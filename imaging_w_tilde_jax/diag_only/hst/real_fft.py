import numpy as np
import jax
import jax.numpy as jnp
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
# FFT PSF curvature kernel
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
    Fhat = jnp.fft.rfft2(images_pad, s=(Fy, Fx))                 # (B, Fy, Fx//2+1)
    out_pad = jnp.fft.irfft2(Fhat * Khat_r[None, :, :], s=(Fy, Fx))  # (B, Fy, Fx), real

    cy, cx = Ky // 2, Kx // 2
    return out_pad[:, cy:cy + Hy, cx:cx + Hx]



# ============================================================
# Curvature matrix builder (UNCHANGED)
# ============================================================

from jax.ops import segment_sum

from jax.ops import segment_sum
from jax import lax
import jax.numpy as jnp

def curvature_matrix_from_psf_preload_rfft_jax(
    inv_noise_var,     # (Hy, Hx) float64
    rows, cols, vals,  # COO mapping arrays
    y_shape: int,
    x_shape: int,
    S: int,
    Khat_r,            # (Fy, Fx//2+1) complex128
    Khat_flip_r,       # (Fy, Fx//2+1) complex128
    Ky: int,
    Kx: int,
    batch_size: int = 32,
):
    inv_noise_var = jnp.asarray(inv_noise_var, dtype=jnp.float64)
    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=jnp.float64)

    M = y_shape * x_shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    def apply_W(Fbatch_flat: jnp.ndarray) -> jnp.ndarray:
        B = Fbatch_flat.shape[1]
        Fimg = Fbatch_flat.T.reshape((B, y_shape, x_shape))
        blurred = rfft_convolve2d_same(Fimg, Khat_r, Ky, Kx, fft_shape)
        weighted = blurred * inv_noise_var[None, :, :]
        back = rfft_convolve2d_same(weighted, Khat_flip_r, Ky, Kx, fft_shape)
        return back.reshape((B, M)).T  # (M,B)

    n_blocks = (S + batch_size - 1) // batch_size
    C0 = jnp.zeros((S, S), dtype=jnp.float64)

    # Precompute a [0..batch_size-1] vector once (static size)
    col_offsets = jnp.arange(batch_size, dtype=jnp.int32)

    def body(block_i, C):
        start = block_i * batch_size  # dynamic scalar, OK

        # IMPORTANT: keep the "in block" test using static width batch_size
        in_block = (cols >= start) & (cols < (start + batch_size))

        bc = jnp.where(in_block, cols - start, 0).astype(jnp.int32)
        v  = jnp.where(in_block, vals, 0.0)

        # Build Fbatch on pixel grid
        F = jnp.zeros((M, batch_size), dtype=jnp.float64)
        F = F.at[rows, bc].add(v)

        # Apply W
        G = apply_W(F)  # (M, batch_size)

        # Accumulate into curvature columns for this block
        contrib = vals[:, None] * G[rows, :]  # (nnz, batch_size)

        # ---- fix: segment over source-pixel index, not cols ----
        # In your earlier diagonal code you did: segment_sum(contrib, cols, num_segments=S)
        # That only makes sense if `cols` are the "left" index of curvature (i.e. same mapper)
        # and you are building full C[:, start:start+B]. Keep it as you had:
        Cblock = segment_sum(contrib, cols, num_segments=S)  # (S, batch_size)

        # Mask out columns beyond S in the last block
        width = jnp.maximum(0, S - start)         # dynamic scalar
        width = jnp.minimum(width, batch_size)    # dynamic scalar in [0, batch_size]
        mask = (col_offsets < width).astype(jnp.float64)  # (batch_size,)
        Cblock = Cblock * mask[None, :]  # (S, batch_size)

        # Update full (S, batch_size) slice; always legal
        C = lax.dynamic_update_slice(C, Cblock, (0, start))
        return C

    C = lax.fori_loop(0, n_blocks, body, C0)
    return 0.5 * (C + C.T)

from functools import partial

def build_curvature_rfft_fn(psf_np: np.ndarray, y_shape: int, x_shape: int):
    """
    Precompute Khat_r and Khat_flip_r once (float64), return a curvature function
    that can be jitted and called repeatedly.
    """
    psf = jnp.asarray(psf_np, dtype=jnp.float64)
    Ky, Kx = psf.shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    Khat_r = precompute_Khat_rfft(psf, fft_shape)
    Khat_flip_r = precompute_Khat_rfft(jnp.flip(psf, axis=(0, 1)), fft_shape)

    # Jit wrapper with static shapes
    curvature_jit = jax.jit(
        partial(curvature_matrix_from_psf_preload_rfft_jax, Khat_r=Khat_r, Khat_flip_r=Khat_flip_r, Ky=Ky, Kx=Kx),
        static_argnames=("y_shape", "x_shape", "S", "batch_size"),
    )
    return curvature_jit


@jax.jit
def pixel_triplets_from_subpixel_arrays_jax(
    pix_indexes_for_sub,          # (M_sub, P)
    pix_weights_for_sub,          # (M_sub, P)
    slim_index_for_sub,           # (M_sub,)
    fft_index_for_masked_pixel,   # (N_unmasked,)
    sub_fraction_slim,            # (N_unmasked,)
):
    """
    Build sparse source→image mapping triplets (rows, cols, vals)
    for a fixed-size interpolation stencil.

    This supports:
      - Rectangular source grids (P = 4, bilinear)
      - Adaptive Delaunay meshes (P = 3, barycentric)
      - Any fixed-stencil interpolation

    Assumptions
    -----------
    - Every subpixel maps to exactly P source pixels
    - All entries in pix_indexes_for_sub are valid
    - No padding, no ragged rows, no masking required
    """

    M_sub, P = pix_indexes_for_sub.shape

    # Each subpixel contributes P mapping entries
    sub_ids = jnp.repeat(jnp.arange(M_sub, dtype=jnp.int32), P)

    # Flatten source indices and interpolation weights
    cols = pix_indexes_for_sub.reshape(-1).astype(jnp.int32)
    vals = pix_weights_for_sub.reshape(-1)

    # subpixel -> slim image pixel
    slim_rows = slim_index_for_sub[sub_ids]

    # slim image pixel -> FFT grid pixel
    rows = fft_index_for_masked_pixel[slim_rows]

    # Apply per-image-pixel subpixel normalization
    vals = vals * sub_fraction_slim[slim_rows]

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
    dataset = dataset.apply_w_tilde()

    y_shape, x_shape = dataset.mask.shape

    inv_noise_var = build_inv_noise_var(dataset.noise_map.native)
    inv_noise_var[dataset.mask] = 0.0
    inv_noise_var = jnp.asarray(inv_noise_var, dtype=jnp.float64)

    # ========================================================
    # LOAD SUBPIXEL OVERSAMPLING ARRAYS (AutoLens outputs)
    # ========================================================

    pix_indexes_for_sub = np.load("pix_indexes_for_sub_slim_index.npy")
    pix_weights_for_sub = np.load("pix_weights_for_sub_slim_index.npy")


    slim_index_for_sub  = np.load("slim_index_for_sub_slim_index.npy")
    sub_fraction_slim        = np.load("sub_fraction.npy")  # shape (M_pix,)

    S = int(np.load("pix_pixels.npy"))

    # ========================================================
    # BUILD SUBPIXEL COO → COLLAPSE TO PIXELS
    # ========================================================

    rows, cols, vals = pixel_triplets_from_subpixel_arrays_jax(
        pix_indexes_for_sub,
        pix_weights_for_sub,
        slim_index_for_sub,
        dataset.w_tilde.fft_index_for_masked_pixel,
        sub_fraction_slim,
    )

    # Put this near where you define curvature_matrix_from_psf_preload_jax (or in main before calling it)

    # ------------------------------------------------------------
    # Then replace your call block with this
    # ------------------------------------------------------------
    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=jnp.float64)

    curv_fn = build_curvature_rfft_fn(psf_np=np.array(dataset.psf.native), y_shape=y_shape, x_shape=x_shape)

    # warm up
    C = curv_fn(inv_noise_var, rows, cols, vals, y_shape=y_shape, x_shape=x_shape, S=S, batch_size=300)
    jax.block_until_ready(C)

    # timed
    import time
    start = time.time()

    C = curv_fn(inv_noise_var, rows, cols, vals, y_shape=y_shape, x_shape=x_shape, S=S, batch_size=300)
    jax.block_until_ready(C)

    print(f"JAX Curvature Matrix Time: {time.time() - start:.2f} seconds")

    print("Curvature matrix computed successfully")
    print("Max:", float(jnp.max(C)), "Mean:", float(jnp.mean(C)))

    curvature_matrix_true = np.load("curvature_matrix.npy")

    # Spot check one row (change index as you like)
    i = 301
    print("true row:", curvature_matrix_true[i, :50])
    print("jax  row:", C[i, :50])

    print(np.max(C))
    print(np.min(C))
    print(np.mean(C))

    print(np.max(curvature_matrix_true))
    print(np.min(curvature_matrix_true))
    print(np.mean(curvature_matrix_true))

    ok = np.allclose(C, curvature_matrix_true, rtol=1e-6, atol=1e-6)
    print("allclose:", ok)
    if not ok:
        diff = np.abs(C - curvature_matrix_true)
        ij = np.unravel_index(np.argmax(diff), diff.shape)
        print("Max abs diff:", float(diff[ij]), "at", ij)
        print("true:", float(curvature_matrix_true[ij]), "jax:", float(C[ij]))
        raise AssertionError("curvature mismatch")

    print("OK: curvature_matrix matches curvature_matrix_true")


if __name__ == "__main__":
    main()
