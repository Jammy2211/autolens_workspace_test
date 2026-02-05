import numpy as np
import jax
import jax.numpy as jnp
from jax.ops import segment_sum
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


# ============================================================
# FFT PSF curvature kernel
# ============================================================

def precompute_Khat(kernel, fft_shape):
    Ky, Kx = kernel.shape
    Fy, Fx = fft_shape
    kernel_pad = jnp.pad(kernel, ((0, Fy-Ky), (0, Fx-Kx)))
    return jnp.fft.fft2(kernel_pad)


def fft_convolve2d_same(images, Khat, Ky, Kx):
    B, Hy, Hx = images.shape
    Fy, Fx = Khat.shape

    images_pad = jnp.pad(images, ((0,0),(0,Fy-Hy),(0,Fx-Hx)))
    out = jnp.fft.ifft2(jnp.fft.fft2(images_pad) * Khat[None,:,:])

    cy, cx = Ky//2, Kx//2
    return jnp.real(out[:, cy:cy+Hy, cx:cx+Hx])


# ============================================================
# Curvature matrix builder (UNCHANGED)
# ============================================================

import jax
import jax.numpy as jnp

def curvature_matrix_from_psf_preload_jax__real_space_conv(
    psf,
    inv_noise_var,
    rows,
    cols,
    vals,
    y_shape,
    x_shape,
    S,
    batch_size=32,
):
    import jax
    import jax.numpy as jnp
    from jax import lax

    psf_flip = jnp.flip(psf, axis=(0,1))
    M = y_shape * x_shape
    Ky, Kx = psf.shape
    fft_shape = (y_shape+Ky-1, x_shape+Kx-1)

    def apply_operator_batch_spatial(Fbatch_flat, psf, inv_noise_var, y_shape, x_shape):
        """
        W = H^T N^{-1} H using *spatial* convolutions.
        Fbatch_flat: (M, B) where M=y_shape*x_shape
        returns:     (M, B)
        """
        B = Fbatch_flat.shape[1]
        Fimg = Fbatch_flat.T.reshape((B, y_shape, x_shape))  # (B, H, W)

        # lax.conv_general_dilated expects NCHW (by default for dimension_numbers below)
        x = Fimg[:, None, :, :]  # (N=B, C=1, H, W)

        # PSF kernel: (out_chan=1, in_chan=1, Ky, Kx)
        k = psf[None, None, :, :]

        # "SAME" padding -> output H,W equals input H,W
        # dimension_numbers specify NCHW + OIHW
        blurred = lax.conv_general_dilated(
            x,
            k,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )  # (B,1,H,W)

        weighted = blurred[:, 0, :, :] * inv_noise_var[None, :, :]  # (B,H,W)

        # Backprojection with flipped PSF (H^T)
        kT = jnp.flip(psf, axis=(0, 1))[None, None, :, :]

        back = lax.conv_general_dilated(
            weighted[:, None, :, :],
            kT,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )  # (B,1,H,W)

        out = back[:, 0, :, :].reshape((B, y_shape * x_shape)).T  # (M,B)
        return out

    C = jnp.zeros((S, S), dtype=jnp.float64)

    for start in range(0, S, batch_size):
        end = min(start+batch_size, S)
        in_block = (cols >= start) & (cols < end)
        bc = cols - start
        bc = jnp.where(in_block, bc, 0)
        v = jnp.where(in_block, vals, 0.0)

        F = jnp.zeros((M, batch_size), dtype=jnp.float64)
        F = F.at[rows, bc].add(v)

        G = apply_operator_batch_spatial(F, psf, inv_noise_var, y_shape, x_shape)

        contrib = vals[:,None] * G[rows,:]
        Cblock = segment_sum(contrib, cols, num_segments=S)

        C = C.at[:, start:end].set(Cblock[:, :end-start])

    return 0.5 * (C + C.T)




@jax.jit
def sparse_triplets_from_subpixel_arrays_jax(
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





# ============================================================
# FULL EXAMPLE
# ============================================================

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
    M_pix = y_shape * x_shape

    psf = jnp.asarray(dataset.psf.native, dtype=jnp.float64)

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

    rows, cols, vals = sparse_triplets_from_subpixel_arrays_jax(
        pix_indexes_for_sub,
        pix_weights_for_sub,
        slim_index_for_sub,
        dataset.sparse_operator.fft_index_for_masked_pixel,
        sub_fraction_slim,
    )


    curvature_matrix_from_psf_preload_jax_jit__real_space_conv = jax.jit(
        curvature_matrix_from_psf_preload_jax__real_space_conv,
        static_argnames=("y_shape", "x_shape", "S", "batch_size"),
    )

    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=jnp.float64)

    C = curvature_matrix_from_psf_preload_jax_jit__real_space_conv(
        psf,
        inv_noise_var,
        rows,
        cols,
        vals,
        y_shape=y_shape,  # must be Python int (static)
        x_shape=x_shape,  # must be Python int (static)
        S=S,  # must be Python int (static)
        batch_size=300,  # must be Python int (static)
    )
    jax.block_until_ready(C)
    print(C[:5, :5])

    import time

    start = time.time()

    # First call triggers compilation; subsequent calls are fast
    C = curvature_matrix_from_psf_preload_jax_jit__real_space_conv(
        psf,
        inv_noise_var,
        rows,
        cols,
        vals,
        y_shape=y_shape,  # must be Python int (static)
        x_shape=x_shape,  # must be Python int (static)
        S=S,  # must be Python int (static)
        batch_size=300,  # must be Python int (static)
    )
    jax.block_until_ready(C)
    print(C[:5, :5])
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
