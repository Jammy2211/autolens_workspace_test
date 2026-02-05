import numpy as np
import jax
import jax.numpy as jnp
import autolens as al

jax.config.update("jax_enable_x64", True)

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
    Fhat = jnp.fft.rfft2(images_pad, s=(Fy, Fx))                 # (B, Fy, Fx//2+1)
    out_pad = jnp.fft.irfft2(Fhat * Khat_r[None, :, :], s=(Fy, Fx))  # (B, Fy, Fx), real

    cy, cx = Ky // 2, Kx // 2
    return out_pad[:, cy:cy + Hy, cx:cx + Hx]


def curvature_matrix_via_w_tilde_from(
        inv_noise_var,
        rows, cols, vals,
        y_shape: int, x_shape: int,
        S: int,
        Khat_r, Khat_flip_r,
        Ky: int, Kx: int,
        batch_size: int = 32,
):
    from jax import lax
    import jax.numpy as jnp
    from jax.ops import segment_sum

    inv_noise_var = jnp.asarray(inv_noise_var, dtype=jnp.float64)
    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=jnp.float64)

    M = y_shape * x_shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    def apply_operator(Fbatch_flat: jnp.ndarray) -> jnp.ndarray:
        B = Fbatch_flat.shape[1]
        Fimg = Fbatch_flat.T.reshape((B, y_shape, x_shape))
        blurred = rfft_convolve2d_same(Fimg, Khat_r, Ky, Kx, fft_shape)
        weighted = blurred * inv_noise_var[None, :, :]
        back = rfft_convolve2d_same(weighted, Khat_flip_r, Ky, Kx, fft_shape)
        return back.reshape((B, M)).T  # (M, B)

    n_blocks = (S + batch_size - 1) // batch_size
    S_pad = n_blocks * batch_size  # <-- key

    C0 = jnp.zeros((S, S_pad), dtype=jnp.float64)
    col_offsets = jnp.arange(batch_size, dtype=jnp.int32)

    def body(block_i, C):
        start = block_i * batch_size

        in_block = (cols >= start) & (cols < (start + batch_size))
        bc = jnp.where(in_block, cols - start, 0).astype(jnp.int32)
        v = jnp.where(in_block, vals, 0.0)

        F = jnp.zeros((M, batch_size), dtype=jnp.float64)
        F = F.at[rows, bc].add(v)

        G = apply_operator(F)  # (M, batch_size)

        contrib = vals[:, None] * G[rows, :]  # (nnz, batch_size)
        Cblock = segment_sum(contrib, cols, num_segments=S)  # (S, batch_size)

        # Mask out unused columns in last block (optional but nice)
        width = jnp.minimum(batch_size, jnp.maximum(0, S - start))
        mask = (col_offsets < width).astype(jnp.float64)
        Cblock = Cblock * mask[None, :]

        # SAFE because C has width S_pad, and start+batch_size <= S_pad always
        C = lax.dynamic_update_slice(C, Cblock, (0, start))
        return C

    C_pad = lax.fori_loop(0, n_blocks, body, C0)
    C = C_pad[:, :S]  # <-- slice back to true width

    return 0.5 * (C + C.T)
from functools import partial

def curvature_matrix_diag_via_psf_weighted_noise_from_func(psf_np: np.ndarray, y_shape: int, x_shape: int):
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
        partial(curvature_matrix_via_w_tilde_from, Khat_r=Khat_r, Khat_flip_r=Khat_flip_r, Ky=Ky, Kx=Kx),
        static_argnames=("y_shape", "x_shape", "S", "batch_size"),
    )
    return curvature_jit


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

    inv_noise_var = inverse_noise_variances_from(dataset.noise_map.native)
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

    rows, cols, vals = sparse_triplets_from(
        pix_indexes_for_sub,
        pix_weights_for_sub,
        slim_index_for_sub,
        dataset.mask.fft_index_for_masked_pixel,
        sub_fraction_slim,
        return_rows_slim=False
    )

    # Put this near where you define curvature_matrix_from_psf_preload_jax (or in main before calling it)

    # ------------------------------------------------------------
    # Then replace your call block with this
    # ------------------------------------------------------------
    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=jnp.float64)

    curvature_matrix_diag_from = curvature_matrix_diag_via_psf_weighted_noise_from_func(psf_np=np.array(dataset.psf.native), y_shape=y_shape, x_shape=x_shape)

    # warm up
    C = curvature_matrix_diag_from(inv_noise_var, rows, cols, vals, y_shape=y_shape, x_shape=x_shape, S=S, batch_size=300)
    jax.block_until_ready(C)

    # timed
    import time
    start = time.time()

    C = curvature_matrix_diag_from(inv_noise_var, rows, cols, vals, y_shape=y_shape, x_shape=x_shape, S=S, batch_size=300)
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
