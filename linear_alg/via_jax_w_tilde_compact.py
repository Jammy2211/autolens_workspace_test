import jax.numpy as jnp
import jax
from pathlib import Path
import numpy as np
import time

import autoarray as aa


import autolens as al

# instrument = "vro"
# instrument = "euclid"
instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

folder = Path("linear_alg") / "arrs" / instrument

w_matrix = np.load(f"{folder}/w_matrix.npy")
w_indexes = np.load(f"{folder}/w_indexes.npy")
w_lengths = np.load(f"{folder}/w_lengths.npy")
curvature_preload = np.load(f"{folder}/curvature_preload.npy")
mapping_matrix = np.load(f"{folder}/mapping_matrix.npy")
curvature_matrix = np.load(f"{folder}/curvature_matrix.npy")
data_to_pix_unique = np.load(f"{folder}/data_to_pix_unique.npy")
data_weights = np.load(f"{folder}/data_weights.npy")
pix_lengths = np.load(f"{folder}/pix_lengths.npy")
pixels = np.load(f"{folder}/pixels.npy")

pixel_scales_dict = {"vro": 0.2, "euclid": 0.1, "hst": 0.05, "hst_up": 0.03, "ao": 0.01}
pixel_scale = pixel_scales_dict[instrument]

dataset_path = Path("dataset") / "imaging" / "instruments" / instrument

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
    over_sample_size_pixelization=4,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=0.3
)


dataset = dataset.apply_mask(mask=mask)


"""
__Shapes__
"""
print(f"Data Shape: {dataset.data.shape_native}")
print(f"PSF Shape: {dataset.psf.shape_native}")
print(f"Mapping Matrix Shape: {mapping_matrix.shape}")
print("Curvature Matrix Shape: ", curvature_matrix.shape)

import numpy as np
import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp
from jax import lax


def make_curvature_fn(
    curvature_indexes: tuple[int, ...], curvature_lengths: tuple[int, ...]
):
    """
    Factory that bakes in static curvature_indexes and curvature_lengths.
    These are known at trace time, so loops can be unrolled.
    """

    indexes = tuple(int(i) for i in curvature_indexes)
    lengths = tuple(int(l) for l in curvature_lengths)
    n_data = len(lengths)

    def curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
        curvature_preload: jnp.ndarray,
        data_to_pix_unique: jnp.ndarray,
        data_weights: jnp.ndarray,
        pix_lengths: jnp.ndarray,
        pix_pixels: int,
    ) -> jnp.ndarray:

        cm = jnp.zeros((pix_pixels, pix_pixels))
        ci = 0  # running index into preload

        # Unrolled over static data_0
        for d0 in range(n_data):
            L = lengths[d0]
            for d1_idx in range(L):
                d1 = indexes[ci]
                w_tilde_value = curvature_preload[ci]

                # vectorized pix0 × pix1 update
                max_len = data_weights.shape[1]  # static
                row_w0 = data_weights[d0]  # shape (max_len,)
                row_sp0 = data_to_pix_unique[d0]  # shape (max_len,)

                L0 = pix_lengths[d0]  # traced scalar

                mask0 = jnp.arange(max_len) < L0
                w0 = row_w0 * mask0
                sp0 = jnp.where(mask0, row_sp0, -1)  # -1 for padding

                L1 = pix_lengths[d1]  # dynamic scalar

                row_w1 = data_weights[d1]  # shape (max_len,)
                row_sp1 = data_to_pix_unique[d1]  # shape (max_len,)

                mask1 = jnp.arange(max_len) < L1

                w1 = row_w1 * mask1
                sp1 = jnp.where(mask1, row_sp1, -1)  # -1 marks invalid entries

                # Outer product of weights
                contrib = w_tilde_value * jnp.outer(w0, w1)

                # Scatter into curvature matrix
                cm = cm.at[(sp0[:, None], sp1[None, :])].add(contrib)

                ci += 1

        # Symmetrize
        cm = (cm + cm.T) - jnp.diag(jnp.diag(cm))
        return cm

    return jax.jit(
        curvature_matrix_via_w_tilde_curvature_preload_imaging_from, static_argnums=(4,)
    )


# jitted_curvature_matrix_via_w_tilde_from = jax.jit(
#     curvature_matrix_via_w_tilde_curvature_preload_imaging_from,
#     static_argnums=(1, 2, 6),
# )

# jitted_curvature_matrix_via_w_tilde_from = curvature_matrix_via_w_tilde_curvature_preload_imaging_from

"""
Precompute functions so compute tile not printed.
"""
curvature_preload = np.asarray(curvature_preload)
w_indexes = tuple(np.asarray(w_indexes, dtype=int).tolist())
w_lengths = tuple(np.asarray(w_lengths, dtype=int).tolist())
data_to_pix_unique = jnp.asarray(data_to_pix_unique)
data_weights = jnp.asarray(data_weights)
pix_lengths = jnp.asarray(pix_lengths)
pixels = int(pixels)

start = time.time()

print("Begun JAX jit compile curvature_matrix w_tilde...")

curv_fn = make_curvature_fn(w_indexes, w_lengths)

print(f"Time JAX jit compile curvature_matrix w_tilde: {time.time() - start}")

start = time.time()

print("Calling function...")

curvature_matrix_w_tilde = curv_fn(
    curvature_preload=curvature_preload,
    data_to_pix_unique=data_to_pix_unique,
    data_weights=data_weights,
    pix_lengths=pix_lengths,
    pix_pixels=int(pixels),
)

print(f"Time JAX call curvature_matrix w_tilde: {time.time() - start}")

# curvature_matrix_w_tilde = jitted_curvature_matrix_via_w_tilde_from(
#     curvature_preload=curvature_preload,
#     curvature_indexes=w_indexes,
#     curvature_lengths=w_lengths,
#     data_to_pix_unique=data_to_pix_unique,
#     data_weights=data_weights,
#     pix_lengths=pix_lengths,
#     pix_pixels=int(pixels),
# )
print(np.max(curvature_matrix_w_tilde))
print(np.min(curvature_matrix_w_tilde))

"""
__Time JIT__
"""
start = time.time()

curvature_matrix_w_tilde = jitted_curvature_matrix_via_w_tilde_from(
    curvature_preload=curvature_preload,
    curvature_indexes=w_indexes,
    curvature_lengths=w_lengths,
    data_to_pix_unique=data_to_pix_unique,
    data_weights=data_weights,
    pix_lengths=pix_lengths,
    pix_pixels=int(pixels),
)
print(np.max(curvature_matrix_w_tilde))
print(np.min(curvature_matrix_w_tilde))

print(f"Time JAX jit curvature_matrix w_tilde: {time.time() - start}")


"""
__Time VMap__
"""
