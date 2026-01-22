import numpy as np
import jax
import jax.numpy as jnp

import autolens as al

jax.config.update("jax_enable_x64", True)

def build_convolved_mapping_matrix_fn(psf_kernel, mask, blurring_mask=None):
    convolve = psf_kernel.convolved_mapping_matrix_from

    # Any non-array args must be captured here (static), not passed to jit.
    @jax.jit
    def fn(mapping_matrix, blurring_mapping_matrix=None):
        return convolve(
            mapping_matrix=mapping_matrix,
            mask=mask,
            blurring_mapping_matrix=blurring_mapping_matrix,
            blurring_mask=blurring_mask,
            xp=jnp,                 # captured module, not an argument
        )

    return fn


def build_curvature_from_mapping_matrix_fn(*, use_float64=True):

    dtype = jnp.float64 if use_float64 else jnp.float32

    @jax.jit
    def fn(mapping_matrix, noise_map_slim):
        mapping_matrix = jnp.asarray(mapping_matrix, dtype=dtype)     # (N_pix, N_src)
        noise_map_slim = jnp.asarray(noise_map_slim, dtype=dtype)     # (N_pix,)

        array = mapping_matrix / noise_map_slim[:, None]
        return array.T @ array

    return fn




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

    # ========================================================
    # LOAD SUBPIXEL OVERSAMPLING ARRAYS (AutoLens outputs)
    # ========================================================

    mapping_matrix = np.load("mapping_matrix.npy")

    # ========================================================
    # BUILD SUBPIXEL COO → COLLAPSE TO PIXELS
    # ========================================================

    convolved_fn = build_convolved_mapping_matrix_fn(dataset.psf, mask)

    operated_mapping_matrix = convolved_fn(mapping_matrix)  # (N_pix, N_src)
    jax.block_until_ready(operated_mapping_matrix)

    curv_mm_fn = build_curvature_from_mapping_matrix_fn(use_float64=True)
    C = curv_mm_fn(operated_mapping_matrix, dataset.noise_map.array)
    m1 = C
    m2 = C.T
    C = jnp.where(m1 != 0, m1, m2)
    jax.block_until_ready(C)

    # timed
    import time
    start = time.time()

    operated_mapping_matrix = convolved_fn(mapping_matrix)  # (N_pix, N_src)
    jax.block_until_ready(operated_mapping_matrix)
    print(f"JAX Curvature Matrix Time: {time.time() - start:.2f} seconds")

    start = time.time()
    C = curv_mm_fn(operated_mapping_matrix, dataset.noise_map.array)
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
