import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.ops import segment_sum
from functools import partial

import autolens as al

jax.config.update("jax_enable_x64", True)

# ============================================================
# Utilities
# ============================================================

def inverse_noise_variances_from(noise):
    inv = np.zeros_like(noise, dtype=np.float64)
    good = np.isfinite(noise) & (noise > 0)
    inv[good] = 1.0 / noise[good] ** 2
    return inv


# ============================================================
# Real FFT PSF kernel (same as your production code)
# ============================================================

def precompute_Khat_rfft(kernel_2d: jnp.ndarray, fft_shape):
    Ky, Kx = kernel_2d.shape
    Fy, Fx = fft_shape
    kernel_pad = jnp.pad(kernel_2d, ((0, Fy - Ky), (0, Fx - Kx)))
    return jnp.fft.rfft2(kernel_pad, s=(Fy, Fx))


def rfft_convolve2d_same(images: jnp.ndarray, Khat_r: jnp.ndarray, Ky: int, Kx: int, fft_shape):
    """
    images: (B, Hy, Hx) real float64
    Khat_r: (Fy, Fx//2+1) complex128
    """
    B, Hy, Hx = images.shape
    Fy, Fx = fft_shape

    images_pad = jnp.pad(images, ((0, 0), (0, Fy - Hy), (0, Fx - Hx)))
    Fhat = jnp.fft.rfft2(images_pad, s=(Fy, Fx))
    out_pad = jnp.fft.irfft2(Fhat * Khat_r[None, :, :], s=(Fy, Fx))

    cy, cx = Ky // 2, Kx // 2
    return out_pad[:, cy:cy + Hy, cx:cx + Hx]


# ============================================================
# Triplets builder (same as your standard)
# ============================================================

@jax.jit
def sparse_triplets_from_subpixel_arrays_jax(
    pix_indexes_for_sub,          # (M_sub, P)
    pix_weights_for_sub,          # (M_sub, P)
    slim_index_for_sub,           # (M_sub,)
    fft_index_for_masked_pixel,   # (M_pix,)
    sub_fraction_slim,            # (M_pix,)
):
    M_sub, P = pix_indexes_for_sub.shape

    sub_ids = jnp.repeat(jnp.arange(M_sub, dtype=jnp.int32), P)

    cols = pix_indexes_for_sub.reshape(-1).astype(jnp.int32)
    vals = pix_weights_for_sub.reshape(-1).astype(jnp.float64)

    slim_rows = slim_index_for_sub[sub_ids].astype(jnp.int32)
    rows = fft_index_for_masked_pixel[slim_rows].astype(jnp.int32)

    vals = vals * sub_fraction_slim[slim_rows].astype(jnp.float64)
    return rows, cols, vals


def curvature_matrix_off_diag_with_light_profiles_via_w_tilde_from(
    curvature_weights_slim,      # (M_pix, n_funcs) = (H B) / noise^2  on slim grid
    fft_index_for_masked_pixel,  # (M_pix,) slim -> rect(flat) indices
    rows, cols, vals,            # triplets for sparse mapper A
    y_shape: int,
    x_shape: int,
    S: int,
    Khat_flip_r,                 # precomputed rfft2(flipped PSF padded)
    Ky: int,
    Kx: int,
):
    """
    Computes: off_diag = A^T [ H^T(curvature_weights_native) ]
    where curvature_weights = (H B) / noise^2 already.
    """
    curvature_weights_slim = jnp.asarray(curvature_weights_slim, dtype=jnp.float64)
    fft_index_for_masked_pixel = jnp.asarray(fft_index_for_masked_pixel, dtype=jnp.int32)

    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=jnp.float64)

    M_pix, n_funcs = curvature_weights_slim.shape
    M_rect = y_shape * x_shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    # 1) scatter slim weights onto rectangular grid (flat)
    grid_flat = jnp.zeros((M_rect, n_funcs), dtype=jnp.float64)
    grid_flat = grid_flat.at[fft_index_for_masked_pixel, :].set(curvature_weights_slim)

    # 2) apply H^T = convolution with flipped PSF (one convolution)
    images = grid_flat.T.reshape((n_funcs, y_shape, x_shape))  # (B=n_funcs, Hy, Hx)
    back_native = rfft_convolve2d_same(images, Khat_flip_r, Ky, Kx, fft_shape)

    # 3) gather at mapper rows
    back_flat = back_native.reshape((n_funcs, M_rect)).T       # (M_rect, n_funcs)
    back_at_rows = back_flat[rows, :]                          # (nnz, n_funcs)

    # 4) accumulate into sparse pixels
    contrib = vals[:, None] * back_at_rows
    off_diag = segment_sum(contrib, cols, num_segments=S)      # (S, n_funcs)
    return off_diag


def build_curvature_matrix_off_diag_with_light_profiles_via_w_tilde_from_func(psf_np: np.ndarray, y_shape: int, x_shape: int):
    psf = jnp.asarray(psf_np, dtype=jnp.float64)
    Ky, Kx = psf.shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    psf_flip = jnp.flip(psf, axis=(0, 1))
    Khat_flip_r = precompute_Khat_rfft(psf_flip, fft_shape)

    fn_jit = jax.jit(
        partial(
            curvature_matrix_off_diag_with_light_profiles_via_w_tilde_from,
            Khat_flip_r=Khat_flip_r,
            Ky=Ky,
            Kx=Kx,
        ),
        static_argnames=("y_shape", "x_shape", "S"),
    )
    return fn_jit



# ============================================================
# End-to-end example
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
    psf_np = np.array(dataset.psf.native, dtype=np.float64)

    # inv noise (for constructing curvature_weights_slim if needed)
    inv_noise_var = inverse_noise_variances_from(np.array(dataset.noise_map.native, dtype=np.float64))
    inv_noise_var[np.array(dataset.mask)] = 0.0

    # -----------------------
    # Load oversampling arrays (shared)
    # -----------------------
    pix_indexes_for_sub = np.load("pix_indexes_for_sub_slim_index.npy").astype(np.int32)
    pix_weights_for_sub = np.load("pix_weights_for_sub_slim_index.npy").astype(np.float64)
    slim_index_for_sub  = np.load("slim_index_for_sub_slim_index.npy").astype(np.int32)
    sub_fraction_slim   = np.load("sub_fraction.npy").astype(np.float64)  # (M_pix,)

    # sparse mapper pixel count
    S = int(np.load("pix_pixels.npy"))

    # -----------------------
    # Build sparse mapper triplets (A)
    # -----------------------
    rows, cols, vals = sparse_triplets_from_subpixel_arrays_jax(
        jnp.asarray(pix_indexes_for_sub, dtype=jnp.int32),
        jnp.asarray(pix_weights_for_sub, dtype=jnp.float64),
        jnp.asarray(slim_index_for_sub, dtype=jnp.int32),
        jnp.asarray(dataset.mask.fft_index_for_masked_pixel, dtype=jnp.int32),
        jnp.asarray(sub_fraction_slim, dtype=jnp.float64),
    )

    # Optional safety filter once on CPU (recommended)
    rows_np = np.array(rows, dtype=np.int32)
    cols_np = np.array(cols, dtype=np.int32)
    vals_np = np.array(vals, dtype=np.float64)

    valid = (rows_np >= 0) & (rows_np < y_shape * x_shape) & (cols_np >= 0) & (cols_np < S)
    rows = jnp.asarray(rows_np[valid], dtype=jnp.int32)
    cols = jnp.asarray(cols_np[valid], dtype=jnp.int32)
    vals = jnp.asarray(vals_np[valid], dtype=jnp.float64)

    # -----------------------
    # Dense linear funcs: curvature_weights_slim = L_slim / noise^2
    # You can load it or build it.
    # -----------------------
    # Option A: load precomputed
    curvature_weights = np.load("curvature_weights.npy").astype(np.float64)  # (M_pix, n_funcs)
    off_diag_true = np.load("off_diag.npy").astype(np.float64)

    # Option B: if you have L_slim (M_pix, n_funcs), do:
    # inv_noise_slim = inv_noise_var[~np.array(dataset.mask)].reshape(-1)  # careful: depends how you store slim
    # curvature_weights_slim = L_slim * inv_noise_slim[:, None]

    curvature_weights_slim = jnp.asarray(curvature_weights, dtype=jnp.float64)

    # Mapping slim->rect(flat) (length M_pix)
    fft_index_for_masked_pixel = jnp.asarray(dataset.mask.fft_index_for_masked_pixel, dtype=jnp.int32)

    # -----------------------
    # Build + run
    # -----------------------
    offdiag_fn = build_curvature_matrix_off_diag_with_light_profiles_via_w_tilde_from_func(psf_np=psf_np, y_shape=y_shape, x_shape=x_shape)

    # warm-up
    off = offdiag_fn(
        curvature_weights_slim,
        fft_index_for_masked_pixel,
        rows, cols, vals,
        y_shape=y_shape,
        x_shape=x_shape,
        S=S,
    )
    jax.block_until_ready(off)

    # timed
    t0 = time.time()
    off = offdiag_fn(
        curvature_weights_slim,
        fft_index_for_masked_pixel,
        rows, cols, vals,
        y_shape=y_shape,
        x_shape=x_shape,
        S=S,
    )
    jax.block_until_ready(off)
    print(f"[JAX] sparse×dense offdiag (triplets, rFFT) time: {time.time() - t0:.3f} sec")

    off_np = np.array(off)
    print("[stats] offdiag shape:", off_np.shape, "max:", off_np.max(), "min:", off_np.min(), "mean:", off_np.mean())

    curvature_matrix_off_diag_true =  np.load("off_diag.npy")

    ok = np.allclose(off, curvature_matrix_off_diag_true, rtol=1e-6, atol=1e-6)
    print("allclose:", ok)
    if not ok:
        diff = np.abs(off - curvature_matrix_off_diag_true)
        ij = np.unravel_index(np.argmax(diff), diff.shape)
        print("Max abs diff:", float(diff[ij]), "at", ij)
        print("true:", float(curvature_matrix_off_diag_true[ij]), "jax:", float(off[ij]))
        raise AssertionError("curvature mismatch")

    print("OK: curvature_matrix matches curvature_matrix_off_diag_true")

if __name__ == "__main__":
    main()
