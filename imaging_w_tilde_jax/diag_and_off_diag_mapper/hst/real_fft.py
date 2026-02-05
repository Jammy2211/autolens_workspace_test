import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.ops import segment_sum
from jax import lax
from functools import partial
import autolens as al

jax.config.update("jax_enable_x64", True)

# ============================================================
# Utilities
# ============================================================

def inverse_noise_variances_from(noise):
    inv = np.zeros_like(noise, dtype=np.float64)
    good = np.isfinite(noise) & (noise > 0)
    inv[good] = 1.0 / noise[good]**2
    return inv


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



def curvature_matrix_off_diag_via_w_tilde_from(
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
    """

    import jax.numpy as jnp
    from jax import lax
    from jax.ops import segment_sum

    inv_noise_var = jnp.asarray(inv_noise_var, dtype=jnp.float64)

    rows0 = jnp.asarray(rows0, dtype=jnp.int32)
    cols0 = jnp.asarray(cols0, dtype=jnp.int32)
    vals0 = jnp.asarray(vals0, dtype=jnp.float64)

    rows1 = jnp.asarray(rows1, dtype=jnp.int32)
    cols1 = jnp.asarray(cols1, dtype=jnp.int32)
    vals1 = jnp.asarray(vals1, dtype=jnp.float64)

    M = y_shape * x_shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    def apply_operator(Fbatch_flat: jnp.ndarray) -> jnp.ndarray:
        B = Fbatch_flat.shape[1]
        Fimg = Fbatch_flat.T.reshape((B, y_shape, x_shape))
        blurred = rfft_convolve2d_same(Fimg, Khat_r, Ky, Kx, fft_shape)
        weighted = blurred * inv_noise_var[None, :, :]
        back = rfft_convolve2d_same(weighted, Khat_flip_r, Ky, Kx, fft_shape)
        return back.reshape((B, M)).T  # (M, B)

    # -----------------------------
    # FIX: pad output width so dynamic_update_slice never clamps
    # -----------------------------
    n_blocks = (S1 + batch_size - 1) // batch_size
    S1_pad = n_blocks * batch_size

    F01_0 = jnp.zeros((S0, S1_pad), dtype=jnp.float64)

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
        Gbatch = apply_operator(Fbatch)  # (M, batch_size)

        # Project with A0^T -> (S0, batch_size)
        contrib = vals0[:, None] * Gbatch[rows0, :]
        block = segment_sum(contrib, cols0, num_segments=S0)  # (S0, batch_size)

        # Mask out columns beyond S1 in the last block
        width = jnp.minimum(batch_size, jnp.maximum(0, S1 - start))
        mask = (col_offsets < width).astype(jnp.float64)
        block = block * mask[None, :]

        # Safe because start+batch_size <= S1_pad always
        F01 = lax.dynamic_update_slice(F01, block, (0, start))
        return F01

    F01_pad = lax.fori_loop(0, n_blocks, body, F01_0)

    # Slice back to true width
    return F01_pad[:, :S1]


def build_curvature_matrix_off_diag_via_w_tilde_from_func(psf_np: np.ndarray, y_shape: int, x_shape: int):
    """
    Matches your diagonal curvature_matrix_diag_via_psf_weighted_noise_from_func:
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
            curvature_matrix_off_diag_via_w_tilde_from,
            Khat_r=Khat_r,
            Khat_flip_r=Khat_flip_r,
            Ky=Ky,
            Kx=Kx,
        ),
        static_argnames=("y_shape", "x_shape", "S0", "S1", "batch_size"),
    )
    return offdiag_jit


def sparse_triplets_from(
    pix_indexes_for_sub,          # (M_sub, P)
    pix_weights_for_sub,          # (M_sub, P)
    slim_index_for_sub,           # (M_sub,)
    fft_index_for_masked_pixel,   # (N_unmasked,)
    sub_fraction_slim,            # (N_unmasked,)
    *,
    return_rows_slim: bool = True,
    xp=np,
):
    """
    Build sparse source→image mapping triplets (rows, cols, vals)
    for a fixed-size interpolation stencil.

    This supports both:
      - NumPy (xp=np)
      - JAX  (xp=jax.numpy)

    Parameters
    ----------
    pix_indexes_for_sub
        Source pixel indices for each subpixel (M_sub, P)
    pix_weights_for_sub
        Interpolation weights for each subpixel (M_sub, P)
    slim_index_for_sub
        Mapping subpixel -> slim image pixel index (M_sub,)
    fft_index_for_masked_pixel
        Mapping slim pixel -> rectangular FFT-grid pixel index (N_unmasked,)
    sub_fraction_slim
        Oversampling normalization per slim pixel (N_unmasked,)
    xp
        Backend module (np or jnp)

    Returns
    -------
    rows : (nnz,) int32
        Rectangular FFT grid row index per mapping entry
    cols : (nnz,) int32
        Source pixel index per mapping entry
    vals : (nnz,) float64
        Mapping weight per entry including sub_fraction normalization
    """
    # ----------------------------
    # NumPy path (HOST)
    # ----------------------------
    if xp is np:
        pix_indexes_for_sub = np.asarray(pix_indexes_for_sub, dtype=np.int32)
        pix_weights_for_sub = np.asarray(pix_weights_for_sub, dtype=np.float64)
        slim_index_for_sub  = np.asarray(slim_index_for_sub,  dtype=np.int32)
        fft_index_for_masked_pixel = np.asarray(fft_index_for_masked_pixel, dtype=np.int32)
        sub_fraction_slim    = np.asarray(sub_fraction_slim,    dtype=np.float64)

        M_sub, P = pix_indexes_for_sub.shape

        sub_ids = np.repeat(np.arange(M_sub, dtype=np.int32), P)  # (M_sub*P,)

        cols = pix_indexes_for_sub.reshape(-1)                    # int32
        vals = pix_weights_for_sub.reshape(-1)                    # float64

        slim_rows = slim_index_for_sub[sub_ids]                   # int32
        vals = vals * sub_fraction_slim[slim_rows]                # float64

        if return_rows_slim:
            return slim_rows, cols, vals

        rows = fft_index_for_masked_pixel[slim_rows]
        return rows, cols, vals

    # ----------------------------
    # JAX path (DEVICE)
    # ----------------------------
    # We intentionally avoid np.asarray anywhere here.
    # Assume xp is jax.numpy (or a compatible array module).
    pix_indexes_for_sub = xp.asarray(pix_indexes_for_sub, dtype=xp.int32)
    pix_weights_for_sub = xp.asarray(pix_weights_for_sub, dtype=xp.float64)
    slim_index_for_sub  = xp.asarray(slim_index_for_sub,  dtype=xp.int32)
    fft_index_for_masked_pixel = xp.asarray(fft_index_for_masked_pixel, dtype=xp.int32)
    sub_fraction_slim    = xp.asarray(sub_fraction_slim,    dtype=xp.float64)

    M_sub, P = pix_indexes_for_sub.shape

    sub_ids = xp.repeat(xp.arange(M_sub, dtype=xp.int32), P)

    cols = pix_indexes_for_sub.reshape(-1)
    vals = pix_weights_for_sub.reshape(-1)

    slim_rows = slim_index_for_sub[sub_ids]
    vals = vals * sub_fraction_slim[slim_rows]

    if return_rows_slim:
        return slim_rows, cols, vals

    rows = fft_index_for_masked_pixel[slim_rows]
    return rows, cols, vals

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
    dataset = dataset.apply_sparse_operator()

    y_shape, x_shape = dataset.mask.shape

    inv_noise_var = inverse_noise_variances_from(np.array(dataset.noise_map.native, dtype=np.float64))
    inv_noise_var[np.array(dataset.mask)] = 0.0
    inv_noise_var = jnp.asarray(inv_noise_var, dtype=jnp.float64)

    fft_index_for_masked_pixel = jnp.asarray(
        dataset.mask.fft_index_for_masked_pixel, dtype=jnp.int32
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
    rows0, cols0, vals0 = sparse_triplets_from(
        pix_indexes_for_sub_0,
        pix_weights_for_sub_0,
        slim_index_for_sub,
        fft_index_for_masked_pixel,
        sub_fraction_slim,
        return_rows_slim=False
    )
    rows1, cols1, vals1 = sparse_triplets_from(
        pix_indexes_for_sub_1,
        pix_weights_for_sub_1,
        slim_index_for_sub,
        fft_index_for_masked_pixel,
        sub_fraction_slim,
        return_rows_slim=False
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
    offdiag_fn = build_curvature_matrix_off_diag_via_w_tilde_from_func(
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
