import numpy as np
import jax
import jax.numpy as jnp
from jax.ops import segment_sum
import autolens as al
from functools import partial

jax.config.update("jax_enable_x64", True)

# ============================================================
# Precision policy
# ============================================================
DTYPE_FFT = jnp.float32   # FFT / convolution
DTYPE_ACC = jnp.float64   # accumulation / curvature

# ============================================================
# Utilities
# ============================================================

def inverse_noise_variances_from(noise):
    inv = np.zeros_like(noise, dtype=np.float64)
    good = np.isfinite(noise) & (noise > 0)
    inv[good] = 1.0 / noise[good]**2
    return inv

# ============================================================
# rFFT PSF curvature kernel (mixed precision)
# ============================================================

def precompute_Khat_rfft(kernel_2d: jnp.ndarray, fft_shape):
    """
    kernel_2d: (Ky, Kx) real
    fft_shape: (Fy, Fx)
    returns: rfft2(padded_kernel) with shape (Fy, Fx//2+1)
             complex64 if kernel is float32
    """
    Ky, Kx = kernel_2d.shape
    Fy, Fx = fft_shape
    kernel_pad = jnp.pad(kernel_2d, ((0, Fy - Ky), (0, Fx - Kx)))
    return jnp.fft.rfft2(kernel_pad, s=(Fy, Fx))

def rfft_convolve2d_same(images: jnp.ndarray, Khat_r: jnp.ndarray, Ky: int, Kx: int, fft_shape):
    """
    Batched real FFT convolution, returning 'same' output.

    images: (B, Hy, Hx) real (float32 here)
    Khat_r: (Fy, Fx//2+1) complex64
    """
    B, Hy, Hx = images.shape
    Fy, Fx = fft_shape

    images_pad = jnp.pad(images, ((0, 0), (0, Fy - Hy), (0, Fx - Hx)))
    Fhat = jnp.fft.rfft2(images_pad, s=(Fy, Fx))  # (B, Fy, Fx//2+1), complex64
    out_pad = jnp.fft.irfft2(Fhat * Khat_r[None, :, :], s=(Fy, Fx))  # (B, Fy, Fx), float32

    cy, cx = Ky // 2, Kx // 2
    return out_pad[:, cy:cy + Hy, cx:cx + Hx]

# ============================================================
# Curvature matrix builder (FFT32, accumulation 64)
# ============================================================

def curvature_matrix_from_psf_preload_rfft_mixed_jax(
    inv_noise_var,     # (Hy,Hx) float32 inside
    rows, cols, vals,  # rows/cols int32; vals float64
    y_shape: int,
    x_shape: int,
    S: int,
    Khat_r,            # complex64
    Khat_flip_r,       # complex64
    Ky: int,
    Kx: int,
    batch_size: int = 32,
):
    """
    W(F) = H^T N^{-1} H(F)

    FFT path in float32, accumulation in float64.
    """

    # Fixed / cheap casts
    inv_noise_var = jnp.asarray(inv_noise_var, dtype=DTYPE_FFT)

    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=DTYPE_ACC)

    M = y_shape * x_shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    def apply_operator(Fbatch_acc: jnp.ndarray) -> jnp.ndarray:
        """
        Fbatch_acc: (M,B) float64
        returns:    (M,B) float32
        """
        B = Fbatch_acc.shape[1]
        Fimg = Fbatch_acc.T.reshape((B, y_shape, x_shape)).astype(DTYPE_FFT)

        blurred = rfft_convolve2d_same(Fimg, Khat_r, Ky, Kx, fft_shape)              # float32
        weighted = blurred * inv_noise_var[None, :, :]                               # float32
        back = rfft_convolve2d_same(weighted, Khat_flip_r, Ky, Kx, fft_shape)        # float32

        return back.reshape((B, M)).T  # (M,B) float32

    C = jnp.zeros((S, S), dtype=DTYPE_ACC)

    for start in range(0, S, batch_size):
        end = min(start + batch_size, S)

        in_block = (cols >= start) & (cols < end)
        bc = jnp.where(in_block, cols - start, 0).astype(jnp.int32)
        v  = jnp.where(in_block, vals, 0.0)  # float64

        # Build RHS in float64 (keeps scatter exact-ish); FFT casts to float32 internally
        F = jnp.zeros((M, batch_size), dtype=DTYPE_ACC)
        F = F.at[rows, bc].add(v)

        G32 = apply_operator(F)               # float32
        G = G32.astype(DTYPE_ACC)      # float64 for accumulation

        contrib = vals[:, None] * G[rows, :]                      # float64
        Cblock = segment_sum(contrib, cols, num_segments=S)       # float64

        C = C.at[:, start:end].set(Cblock[:, :end - start])

    return 0.5 * (C + C.T)

# ============================================================
# Build function with precomputed kernels (FFT32)
# ============================================================

def build_curvature_rfft_mixed_fn(psf_np: np.ndarray, y_shape: int, x_shape: int):
    """
    Precompute Khat_r and Khat_flip_r in float32 (complex64).
    Return jitted curvature function (accumulation float64).
    """
    psf32 = jnp.asarray(psf_np, dtype=DTYPE_FFT)
    Ky, Kx = psf32.shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    Khat_r = precompute_Khat_rfft(psf32, fft_shape)                       # complex64
    Khat_flip_r = precompute_Khat_rfft(jnp.flip(psf32, axis=(0, 1)), fft_shape)

    curvature_jit = jax.jit(
        partial(
            curvature_matrix_from_psf_preload_rfft_mixed_jax,
            Khat_r=Khat_r,
            Khat_flip_r=Khat_flip_r,
            Ky=Ky,
            Kx=Kx,
        ),
        static_argnames=("y_shape", "x_shape", "S", "batch_size"),
    )
    return curvature_jit

# ============================================================
# Triplets builder (keep vals float64, indices int32)
# ============================================================

@jax.jit
def sparse_triplets_from_subpixel_arrays_jax(
    pix_indexes_for_sub,          # (M_sub, P)
    pix_weights_for_sub,          # (M_sub, P)
    slim_index_for_sub,           # (M_sub,)
    fft_index_for_masked_pixel,   # (N_unmasked,)
    sub_fraction_slim,            # (N_unmasked,)
):
    M_sub, P = pix_indexes_for_sub.shape
    sub_ids = jnp.repeat(jnp.arange(M_sub, dtype=jnp.int32), P)

    cols = pix_indexes_for_sub.reshape(-1).astype(jnp.int32)
    vals = pix_weights_for_sub.reshape(-1).astype(DTYPE_ACC)

    slim_rows = slim_index_for_sub[sub_ids]
    rows = fft_index_for_masked_pixel[slim_rows].astype(jnp.int32)

    vals = vals * sub_fraction_slim[slim_rows].astype(DTYPE_ACC)
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

    # Noise inverse variance: compute in float64 then cast to float32 for FFT path
    inv_noise_var = inverse_noise_variances_from(np.array(dataset.noise_map.native, dtype=np.float64))
    inv_noise_var[np.array(dataset.mask)] = 0.0
    inv_noise_var = jnp.asarray(inv_noise_var, dtype=DTYPE_FFT)

    # ========================================================
    # LOAD SUBPIXEL OVERSAMPLING ARRAYS
    # ========================================================
    pix_indexes_for_sub = np.load("pix_indexes_for_sub_slim_index.npy")
    pix_weights_for_sub = np.load("pix_weights_for_sub_slim_index.npy")
    slim_index_for_sub  = np.load("slim_index_for_sub_slim_index.npy")
    sub_fraction_slim   = np.load("sub_fraction.npy")  # shape (M_pix,)
    S = int(np.load("pix_pixels.npy"))

    rows, cols, vals = sparse_triplets_from_subpixel_arrays_jax(
        jnp.asarray(pix_indexes_for_sub, dtype=jnp.int32),
        jnp.asarray(pix_weights_for_sub, dtype=DTYPE_ACC),
        jnp.asarray(slim_index_for_sub, dtype=jnp.int32),
        jnp.asarray(dataset.sparse_operator.fft_index_for_masked_pixel, dtype=jnp.int32),
        jnp.asarray(sub_fraction_slim, dtype=DTYPE_ACC),
    )

    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=DTYPE_ACC)

    # Build mixed-precision curvature fn (Khat precomputed in float32)
    curvature_matrix_diag_from = build_curvature_rfft_mixed_fn(
        psf_np=np.array(dataset.psf.native, dtype=np.float32),  # PSF stored as float32 for FFT speed
        y_shape=y_shape,
        x_shape=x_shape,
    )

    # warm up
    C = curvature_matrix_diag_from(inv_noise_var, rows, cols, vals, y_shape=y_shape, x_shape=x_shape, S=S, batch_size=300)
    jax.block_until_ready(C)

    # timed
    import time
    start = time.time()
    C = curvature_matrix_diag_from(inv_noise_var, rows, cols, vals, y_shape=y_shape, x_shape=x_shape, S=S, batch_size=300)
    jax.block_until_ready(C)
    print(f"JAX Curvature Matrix Time (mixed): {time.time() - start:.3f} seconds")

    # Compare to truth
    curvature_matrix_true = np.load("curvature_matrix.npy")
    C_np = np.array(C)

    ok = np.allclose(C_np, curvature_matrix_true, rtol=1e-6, atol=1e-6)
    print("allclose:", ok)
    if not ok:
        diff = np.abs(C_np - curvature_matrix_true)
        ij = np.unravel_index(np.argmax(diff), diff.shape)
        print("Max abs diff:", float(diff[ij]), "at", ij)
        print("true:", float(curvature_matrix_true[ij]), "jax:", float(C_np[ij]))

    print("Max:", float(C.max()), "Mean:", float(C.mean()))

if __name__ == "__main__":
    main()
