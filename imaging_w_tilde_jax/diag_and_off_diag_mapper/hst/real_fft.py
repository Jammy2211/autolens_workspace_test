import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.ops import segment_sum
from jax import lax
import autolens as al

jax.config.update("jax_enable_x64", True)

# ============================================================
# Utilities
# ============================================================

def build_inv_noise_var(noise):
    inv = np.zeros_like(noise, dtype=np.float64)
    good = np.isfinite(noise) & (noise > 0)
    inv[good] = 1.0 / noise[good]**2
    return inv


# ============================================================
# Real FFT PSF curvature kernel (MATCHES YOUR DIAGONAL API)
# ============================================================

def precompute_Khat_rfft(kernel_2d: jnp.ndarray, fft_shape):
    """
    kernel_2d: (Ky, Kx) real
    fft_shape: (Fy, Fx) where Fy = Hy+Ky-1, Fx = Hx+Kx-1
    returns: rfft2(padded_kernel) with shape (Fy, Fx//2+1), complex128 if input float64
    """
    Ky, Kx = kernel_2d.shape
    Fy, Fx = fft_shape
    kernel_pad = jnp.pad(kernel_2d, ((0, Fy - Ky), (0, Fx - Kx)))
    return jnp.fft.rfft2(kernel_pad, s=(Fy, Fx))


def rfft_convolve2d_same(images: jnp.ndarray, Khat_r: jnp.ndarray, Ky: int, Kx: int, fft_shape):
    """
    Batched real FFT convolution, returning 'same' output.

    images: (B, Hy, Hx) real float64
    Khat_r: (Fy, Fx//2+1) complex128  (rfft2 of padded kernel)
    fft_shape: (Fy, Fx) must equal (Hy+Ky-1, Hx+Kx-1)
    """
    B, Hy, Hx = images.shape
    Fy, Fx = fft_shape

    images_pad = jnp.pad(images, ((0, 0), (0, Fy - Hy), (0, Fx - Hx)))
    Fhat = jnp.fft.rfft2(images_pad, s=(Fy, Fx))                   # (B, Fy, Fx//2+1)
    out_pad = jnp.fft.irfft2(Fhat * Khat_r[None, :, :], s=(Fy, Fx)) # (B, Fy, Fx), real

    cy, cx = Ky // 2, Kx // 2
    return out_pad[:, cy:cy + Hy, cx:cx + Hx]



# ============================================================
# Off-diagonal curvature using SAME real-FFT operator + SAME API STYLE
# ============================================================

def curvature_matrix_offdiag_from_psf_preload_rfft_jax(
    inv_noise_var,     # (Hy, Hx) float64
    rows0, cols0, vals0,
    rows1, cols1, vals1,
    y_shape: int,
    x_shape: int,
    S0: int,
    S1: int,
    Khat_r,            # rfft2(psf padded)
    Khat_flip_r,       # rfft2(flipped psf padded)
    Ky: int,
    Kx: int,
    batch_size: int = 32,
):
    """
    Off-diagonal curvature block:
        F01 = A0^T W A1
    Returns: (S0, S1)

    This is the exact off-diagonal analogue of your diagonal builder:
      - Build RHS F = A1[:, block] via (rows1, cols1, vals1)
      - Apply W via real FFT with (Khat_r, Khat_flip_r)
      - Project with A0^T using (rows0, cols0, vals0) and segment_sum over cols0
    """
    inv_noise_var = jnp.asarray(inv_noise_var, dtype=jnp.float64)

    rows0 = jnp.asarray(rows0, dtype=jnp.int32)
    cols0 = jnp.asarray(cols0, dtype=jnp.int32)
    vals0 = jnp.asarray(vals0, dtype=jnp.float64)

    rows1 = jnp.asarray(rows1, dtype=jnp.int32)
    cols1 = jnp.asarray(cols1, dtype=jnp.int32)
    vals1 = jnp.asarray(vals1, dtype=jnp.float64)

    M = y_shape * x_shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    def apply_W(Fbatch_flat: jnp.ndarray) -> jnp.ndarray:
        B = Fbatch_flat.shape[1]
        Fimg = Fbatch_flat.T.reshape((B, y_shape, x_shape))
        blurred = rfft_convolve2d_same(Fimg, Khat_r, Ky, Kx, fft_shape)
        weighted = blurred * inv_noise_var[None, :, :]
        back = rfft_convolve2d_same(weighted, Khat_flip_r, Ky, Kx, fft_shape)
        return back.reshape((B, M)).T  # (M,B)

    n_blocks = (S1 + batch_size - 1) // batch_size
    F01_0 = jnp.zeros((S0, S1), dtype=jnp.float64)

    col_offsets = jnp.arange(batch_size, dtype=jnp.int32)

    def body(block_i, F01):
        start = block_i * batch_size

        # Select mapper-1 entries in this column block
        in_block = (cols1 >= start) & (cols1 < (start + batch_size))
        bc = jnp.where(in_block, cols1 - start, 0).astype(jnp.int32)
        v  = jnp.where(in_block, vals1, 0.0)

        # Assemble RHS block: (M, batch_size)
        Fbatch = jnp.zeros((M, batch_size), dtype=jnp.float64)
        Fbatch = Fbatch.at[rows1, bc].add(v)

        # Apply W
        Gbatch = apply_W(Fbatch)  # (M, batch_size)

        # Project with A0^T -> (S0, batch_size)
        contrib = vals0[:, None] * Gbatch[rows0, :]
        block = segment_sum(contrib, cols0, num_segments=S0)  # (S0, batch_size)

        # Mask out columns beyond S1 in last block (static update shape)
        width = jnp.maximum(0, S1 - start)
        width = jnp.minimum(width, batch_size)
        mask = (col_offsets < width).astype(jnp.float64)
        block = block * mask[None, :]

        F01 = lax.dynamic_update_slice(F01, block, (0, start))
        return F01

    F01 = lax.fori_loop(0, n_blocks, body, F01_0)
    return F01


def build_offdiag_rfft_fn(psf_np: np.ndarray, y_shape: int, x_shape: int):
    """
    Matches your diagonal build_curvature_rfft_fn:
      - precomputes Khat_r and Khat_flip_r once
      - returns a jitted function with the SAME static args pattern
    """
    psf = jnp.asarray(psf_np, dtype=jnp.float64)
    Ky, Kx = psf.shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    Khat_r = precompute_Khat_rfft(psf, fft_shape)
    Khat_flip_r = precompute_Khat_rfft(jnp.flip(psf, axis=(0, 1)), fft_shape)

    offdiag_jit = jax.jit(
        partial(
            curvature_matrix_offdiag_from_psf_preload_rfft_jax,
            Khat_r=Khat_r,
            Khat_flip_r=Khat_flip_r,
            Ky=Ky,
            Kx=Kx,
        ),
        static_argnames=("y_shape", "x_shape", "S0", "S1", "batch_size"),
    )
    return offdiag_jit


# ============================================================
# End-to-end test: loads mapper 0 + mapper 1 arrays, computes F01
# ============================================================

from functools import partial

def main():
    dataset = al.Imaging.from_fits(
        data_path="data.fits",
        psf_path="psf.fits",
        noise_map_path="noise_map.fits",
        pixel_scales=0.05,
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=3.5,
    )

    dataset = dataset.apply_mask(mask)
    dataset = dataset.apply_w_tilde()

    y_shape, x_shape = dataset.mask.shape

    inv_noise_var = build_inv_noise_var(np.array(dataset.noise_map.native, dtype=np.float64))
    inv_noise_var[np.array(dataset.mask)] = 0.0
    inv_noise_var = jnp.asarray(inv_noise_var, dtype=jnp.float64)

    fft_index_for_masked_pixel = jnp.asarray(
        dataset.w_tilde.fft_index_for_masked_pixel, dtype=jnp.int32
    )

    # Shared oversampling arrays
    slim_index_for_sub = jnp.asarray(
        np.load("slim_index_for_sub_slim_index.npy").astype(np.int32),
        dtype=jnp.int32,
    )
    sub_fraction_slim = jnp.asarray(
        np.load("sub_fraction.npy").astype(np.float64),
        dtype=jnp.float64,
    )

    # Mapper 0 arrays
    pix_indexes_for_sub_0 = jnp.asarray(
        np.load("pix_indexes_for_sub_slim_index_0.npy").astype(np.int32),
        dtype=jnp.int32,
    )
    pix_weights_for_sub_0 = jnp.asarray(
        np.load("pix_weights_for_sub_slim_index_0.npy").astype(np.float64),
        dtype=jnp.float64,
    )
    S0 = int(np.load("pix_pixels.npy"))

    # Mapper 1 arrays
    pix_indexes_for_sub_1 = jnp.asarray(
        np.load("pix_indexes_for_sub_slim_index_1.npy").astype(np.int32),
        dtype=jnp.int32,
    )
    pix_weights_for_sub_1 = jnp.asarray(
        np.load("pix_weights_for_sub_slim_index_1.npy").astype(np.float64),
        dtype=jnp.float64,
    )
    S1 = S0  # adjust if your mapper_1 has different number of pixels

    print(f"[shapes] image=({y_shape},{x_shape})  S0={S0}  S1={S1}")

    # Build COO triplets
    rows0, cols0, vals0 = pixel_triplets_from_subpixel_arrays_jax(
        pix_indexes_for_sub_0,
        pix_weights_for_sub_0,
        slim_index_for_sub,
        fft_index_for_masked_pixel,
        sub_fraction_slim,
    )
    rows1, cols1, vals1 = pixel_triplets_from_subpixel_arrays_jax(
        pix_indexes_for_sub_1,
        pix_weights_for_sub_1,
        slim_index_for_sub,
        fft_index_for_masked_pixel,
        sub_fraction_slim,
    )

    # Optional safety filter (recommended in production if indices can ever drift)
    # It’s cheap and prevents silent bad indices from corrupting results.
    rows0_np, cols0_np, vals0_np = np.array(rows0), np.array(cols0), np.array(vals0)
    rows1_np, cols1_np, vals1_np = np.array(rows1), np.array(cols1), np.array(vals1)

    valid0 = (rows0_np >= 0) & (rows0_np < y_shape * x_shape) & (cols0_np >= 0) & (cols0_np < S0)
    valid1 = (rows1_np >= 0) & (rows1_np < y_shape * x_shape) & (cols1_np >= 0) & (cols1_np < S1)

    rows0 = jnp.asarray(rows0_np[valid0], dtype=jnp.int32)
    cols0 = jnp.asarray(cols0_np[valid0], dtype=jnp.int32)
    vals0 = jnp.asarray(vals0_np[valid0], dtype=jnp.float64)

    rows1 = jnp.asarray(rows1_np[valid1], dtype=jnp.int32)
    cols1 = jnp.asarray(cols1_np[valid1], dtype=jnp.int32)
    vals1 = jnp.asarray(vals1_np[valid1], dtype=jnp.float64)

    # Build off-diagonal fn (matches diagonal API: precompute Khat, jit once)
    offdiag_fn = build_offdiag_rfft_fn(
        psf_np=np.array(dataset.psf.native, dtype=np.float64),
        y_shape=y_shape,
        x_shape=x_shape,
    )

    # Warm-up
    F01 = offdiag_fn(
        inv_noise_var,
        rows0, cols0, vals0,
        rows1, cols1, vals1,
        y_shape=y_shape,
        x_shape=x_shape,
        S0=S0,
        S1=S1,
        batch_size=300,
    )
    jax.block_until_ready(F01)

    # Timed
    t0 = time.time()
    F01 = offdiag_fn(
        inv_noise_var,
        rows0, cols0, vals0,
        rows1, cols1, vals1,
        y_shape=y_shape,
        x_shape=x_shape,
        S0=S0,
        S1=S1,
        batch_size=300,
    )
    jax.block_until_ready(F01)
    print(f"[JAX] off-diagonal time (rFFT): {time.time() - t0:.3f} sec")

    print("[stats] F01 max", float(jnp.max(F01)), "min", float(jnp.min(F01)), "mean", float(jnp.mean(F01)))

    curvature_matrix_true = np.load("curvature_matrix.npy")
    curvature_matrix_off_diagtrue = curvature_matrix_true[:S0, S0:S0 + S1]

    ok = np.allclose(np.array(F01), curvature_matrix_off_diagtrue, rtol=1e-6, atol=1e-6)
    print("[compare] allclose:", ok)
    if not ok:
        diff = np.abs(np.array(F01) - curvature_matrix_off_diagtrue)
        ij = np.unravel_index(np.argmax(diff), diff.shape)
        print("Max abs diff:", float(diff[ij]), "at", ij)
        print("true:", float(curvature_matrix_off_diagtrue[ij]), "jax:", float(np.array(F01)[ij]))
        raise AssertionError("off-diagonal curvature mismatch")

    print("OK: off-diagonal curvature matches truth")


if __name__ == "__main__":
    main()
