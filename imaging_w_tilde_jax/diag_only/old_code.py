


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
    """
    Same math as your FFT version, but uses rfft2/irfft2 and precomputed kernels.

    W(F) = H^T N^{-1} H(F)
    """

    inv_noise_var = jnp.asarray(inv_noise_var, dtype=jnp.float64)

    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=jnp.float64)

    M = y_shape * x_shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    def apply_operator(Fbatch_flat: jnp.ndarray) -> jnp.ndarray:
        B = Fbatch_flat.shape[1]
        Fimg = Fbatch_flat.T.reshape((B, y_shape, x_shape))  # (B,Hy,Hx)

        blurred = rfft_convolve2d_same(Fimg, Khat_r, Ky, Kx, fft_shape)
        weighted = blurred * inv_noise_var[None, :, :]
        back = rfft_convolve2d_same(weighted, Khat_flip_r, Ky, Kx, fft_shape)

        return back.reshape((B, M)).T  # (M,B)

    C = jnp.zeros((S, S), dtype=jnp.float64)

    for start in range(0, S, batch_size):
        end = min(start + batch_size, S)

        in_block = (cols >= start) & (cols < end)
        bc = jnp.where(in_block, cols - start, 0).astype(jnp.int32)
        v  = jnp.where(in_block, vals, 0.0)

        F = jnp.zeros((M, batch_size), dtype=jnp.float64)
        F = F.at[rows, bc].add(v)

        G = apply_operator(F)
        contrib = vals[:, None] * G[rows, :]
        Cblock = segment_sum(contrib, cols, num_segments=S)

        C = C.at[:, start:end].set(Cblock[:, :end - start])

    return 0.5 * (C + C.T)