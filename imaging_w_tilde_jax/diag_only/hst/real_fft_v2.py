import numpy as np
import jax
import jax.numpy as jnp
import autolens as al

jax.config.update("jax_enable_x64", True)

# =============================================================================
# Mixed precision knobs
# =============================================================================
# Use FP32 for the FFT/operator path (fast on most GPUs), but accumulate curvature in FP64.
OP_DTYPE = jnp.float32          # operator / FFT dtype
ACC_DTYPE = jnp.float64         # accumulation dtype for C
K_DTYPE = jnp.float32           # PSF / Khat dtype (complex64 result)
INVNOISE_DTYPE = jnp.float32    # inv_noise_var inside operator


def inverse_noise_variances_from(noise):
    inv = np.zeros_like(noise, dtype=np.float64)
    good = np.isfinite(noise) & (noise > 0)
    inv[good] = 1.0 / noise[good] ** 2
    return inv


def precompute_Khat_rfft(kernel_2d: jnp.ndarray, fft_shape, *, dtype):
    """
    kernel_2d: (Ky, Kx) real
    fft_shape: (Fy, Fx)
    returns: rfft2(padded_kernel) complex64 if dtype=float32, complex128 if float64
    """
    kernel_2d = jnp.asarray(kernel_2d, dtype=dtype)
    Ky, Kx = kernel_2d.shape
    Fy, Fx = fft_shape
    kernel_pad = jnp.pad(kernel_2d, ((0, Fy - Ky), (0, Fx - Kx)))
    return jnp.fft.rfft2(kernel_pad, s=(Fy, Fx))


def rfft_convolve2d_same(images: jnp.ndarray, Khat_r: jnp.ndarray, Ky: int, Kx: int, fft_shape):
    """
    Batched real FFT convolution, returning 'same' output.
    Works for float32/float64; output dtype follows images dtype.
    """
    B, Hy, Hx = images.shape
    Fy, Fx = fft_shape

    images_pad = jnp.pad(images, ((0, 0), (0, Fy - Hy), (0, Fx - Hx)))
    Fhat = jnp.fft.rfft2(images_pad, s=(Fy, Fx))
    out_pad = jnp.fft.irfft2(Fhat * Khat_r[None, :, :], s=(Fy, Fx))

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
    """
    Mixed precision:
      - Build F in OP_DTYPE (float32)
      - Apply FFT operator in OP_DTYPE
      - Build contrib in OP_DTYPE then cast to ACC_DTYPE before segment_sum
      - Accumulate C in ACC_DTYPE (float64)
    """
    from jax import lax
    from jax.ops import segment_sum

    # Indices stay integer
    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)

    # Keep original vals as float64 on input if you like, but compute path uses OP_DTYPE
    vals64 = jnp.asarray(vals, dtype=jnp.float64)
    vals_op = vals64.astype(OP_DTYPE)

    # Operator uses float32 inv_noise
    inv_noise_op = jnp.asarray(inv_noise_var, dtype=INVNOISE_DTYPE)

    M = y_shape * x_shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    def apply_operator(Fbatch_flat: jnp.ndarray) -> jnp.ndarray:
        B = Fbatch_flat.shape[1]
        Fimg = Fbatch_flat.T.reshape((B, y_shape, x_shape))
        blurred = rfft_convolve2d_same(Fimg, Khat_r, Ky, Kx, fft_shape)
        weighted = blurred * inv_noise_op[None, :, :]
        back = rfft_convolve2d_same(weighted, Khat_flip_r, Ky, Kx, fft_shape)
        return back.reshape((B, M)).T  # (M, B) in OP_DTYPE

    n_blocks = (S + batch_size - 1) // batch_size
    S_pad = n_blocks * batch_size

    C0 = jnp.zeros((S, S_pad), dtype=ACC_DTYPE)
    col_offsets = jnp.arange(batch_size, dtype=jnp.int32)

    def body(block_i, C):
        start = block_i * batch_size

        in_block = (cols >= start) & (cols < (start + batch_size))
        bc = jnp.where(in_block, cols - start, 0).astype(jnp.int32)
        v = jnp.where(in_block, vals_op, OP_DTYPE(0.0))

        # (1) scatter in float32
        F = jnp.zeros((M, batch_size), dtype=OP_DTYPE)
        F = F.at[rows, bc].add(v)

        # (2) FFT operator in float32
        G = apply_operator(F)  # (M, batch_size) float32

        # (3) contrib in float32, then cast to float64 before reduce
        # NOTE: use vals_op (float32) for multiplication; you can also use vals64 for extra safety.
        contrib = (vals_op[:, None] * G[rows, :]).astype(ACC_DTYPE)  # (nnz, batch_size) float64

        Cblock = segment_sum(contrib, cols, num_segments=S)  # (S, batch_size) float64

        width = jnp.minimum(batch_size, jnp.maximum(0, S - start))
        mask = (col_offsets < width).astype(ACC_DTYPE)
        Cblock = Cblock * mask[None, :]

        C = lax.dynamic_update_slice(C, Cblock, (0, start))
        return C

    C_pad = lax.fori_loop(0, n_blocks, body, C0)
    C = C_pad[:, :S]
    return 0.5 * (C + C.T)


from functools import partial


def curvature_matrix_diag_via_psf_weighted_noise_from_func(psf_np: np.ndarray, y_shape: int, x_shape: int):
    """
    Precompute Khat_r and Khat_flip_r in float32 (complex64), but keep final C float64.
    """
    psf = jnp.asarray(psf_np, dtype=K_DTYPE)
    Ky, Kx = psf.shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    Khat_r = precompute_Khat_rfft(psf, fft_shape, dtype=K_DTYPE)
    Khat_flip_r = precompute_Khat_rfft(jnp.flip(psf, axis=(0, 1)), fft_shape, dtype=K_DTYPE)

    curvature_jit = jax.jit(
        partial(
            curvature_matrix_via_w_tilde_from,
            Khat_r=Khat_r,
            Khat_flip_r=Khat_flip_r,
            Ky=Ky,
            Kx=Kx,
        ),
        static_argnames=("y_shape", "x_shape", "S", "batch_size"),
    )
    return curvature_jit


def sparse_triplets_from(
    pix_indexes_for_sub,
    pix_weights_for_sub,
    slim_index_for_sub,
    fft_index_for_masked_pixel,
    sub_fraction_slim,
    *,
    return_rows_slim: bool = True,
    xp=np,
):
    if xp is np:
        pix_indexes_for_sub = np.asarray(pix_indexes_for_sub, dtype=np.int32)
        pix_weights_for_sub = np.asarray(pix_weights_for_sub, dtype=np.float64)
        slim_index_for_sub  = np.asarray(slim_index_for_sub,  dtype=np.int32)
        fft_index_for_masked_pixel = np.asarray(fft_index_for_masked_pixel, dtype=np.int32)
        sub_fraction_slim    = np.asarray(sub_fraction_slim,    dtype=np.float64)

        M_sub, P = pix_indexes_for_sub.shape
        sub_ids = np.repeat(np.arange(M_sub, dtype=np.int32), P)

        cols = pix_indexes_for_sub.reshape(-1)
        vals = pix_weights_for_sub.reshape(-1)

        slim_rows = slim_index_for_sub[sub_ids]
        vals = vals * sub_fraction_slim[slim_rows]

        if return_rows_slim:
            return slim_rows.astype(np.int32), cols.astype(np.int32), vals.astype(np.float64)

        rows = fft_index_for_masked_pixel[slim_rows]
        return rows.astype(np.int32), cols.astype(np.int32), vals.astype(np.float64)

    raise NotImplementedError("This example builds triplets on host (NumPy).")


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
    inv_noise_var = jnp.asarray(inv_noise_var, dtype=jnp.float64)  # keep on input; cast inside

    pix_indexes_for_sub = np.load("pix_indexes_for_sub_slim_index.npy")
    pix_weights_for_sub = np.load("pix_weights_for_sub_slim_index.npy")
    slim_index_for_sub  = np.load("slim_index_for_sub_slim_index.npy")
    sub_fraction_slim   = np.load("sub_fraction.npy")
    S = int(np.load("pix_pixels.npy"))

    rows, cols, vals = sparse_triplets_from(
        pix_indexes_for_sub,
        pix_weights_for_sub,
        slim_index_for_sub,
        dataset.mask.fft_index_for_masked_pixel,
        sub_fraction_slim,
        return_rows_slim=False,
        xp=np,
    )

    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=jnp.float64)  # keep float64 for input fidelity

    curvature_matrix_diag_from = curvature_matrix_diag_via_psf_weighted_noise_from_func(
        psf_np=np.array(dataset.psf.native),
        y_shape=y_shape,
        x_shape=x_shape,
    )

    # warm up
    C = curvature_matrix_diag_from(inv_noise_var, rows, cols, vals, y_shape=y_shape, x_shape=x_shape, S=S, batch_size=512)
    jax.block_until_ready(C)

    # timed
    import time
    t0 = time.time()
    C = curvature_matrix_diag_from(inv_noise_var, rows, cols, vals, y_shape=y_shape, x_shape=x_shape, S=S, batch_size=512)
    jax.block_until_ready(C)
    print(f"JAX Curvature Matrix Time (mixed precision): {time.time() - t0:.4f} seconds")

    print("Curvature matrix computed successfully")
    print("Max:", float(jnp.max(C)), "Mean:", float(jnp.mean(C)))

    curvature_matrix_true = np.load("curvature_matrix.npy")
    ok = np.allclose(np.asarray(C), curvature_matrix_true, rtol=1e-6, atol=1e-6)
    print("allclose:", ok)
    if not ok:
        diff = np.abs(np.asarray(C) - curvature_matrix_true)
        ij = np.unravel_index(np.argmax(diff), diff.shape)
        print("Max abs diff:", float(diff[ij]), "at", ij)
        print("true:", float(curvature_matrix_true[ij]), "jax:", float(np.asarray(C)[ij]))
        raise AssertionError("curvature mismatch")

    print("OK: curvature_matrix matches curvature_matrix_true")


if __name__ == "__main__":
    main()
