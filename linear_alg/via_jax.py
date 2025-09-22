import jax.numpy as jnp
import jax
from pathlib import Path
import numpy as np
import time

import autoarray as aa

# instrument = "vro"
# instrument = "euclid"
instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

folder = Path("linear_alg") / "arrs" / instrument

# pix_indexes_for_sub_slim_index = np.load(f"{folder}/pix_indexes_for_sub_slim_index.npy")
# pix_size_for_sub_slim_index = np.load(f"{folder}/pix_size_for_sub_slim_index.npy")
# pix_weights_for_sub_slim_index = np.load(f"{folder}/pix_weights_for_sub_slim_index.npy")
# pixels = np.load(f"{folder}/pixels.npy")
# total_mask_pixels = np.load(f"{folder}/total_mask_pixels.npy")
# slim_index_for_sub_slim_index = np.load(f"{folder}/slim_index_for_sub_slim_index.npy")
# sub_fraction = np.load(f"{folder}/sub_fraction.npy")
# native_index_for_slim_index = np.load(f"{folder}/native_index_for_slim_index.npy")
w_matrix = np.load(f"{folder}/w_matrix.npy")
# psf_operator_matrix_dense = np.load(f"{folder}/psf_operator_matrix_dense.npy")
mapping_matrix = np.load(f"{folder}/mapping_mattrix.npy")
# blurred_mapping_matrix = np.load(f"{folder}/blurred_mapping_mattrix.npy")
# w_tilde_data = np.load(f"{folder}/w_tilde_data.npy")
curvature_matrix = np.load(f"{folder}/curvature_matrix.npy")
# regularization_matrix = np.load(f"{folder}/regularization_matrix.npy")
# data_vector = np.load(f"{folder}/data_vector.npy")
# reconstruction = np.load(f"{folder}/reconstruction.npy")
# mapped_reconstructed_image = np.load(f"{folder}/mapped_reconstructed_image.npy")
# log_evidence = np.load(f"{folder}/log_evidence.npy")

pixel_scales_dict = {"vro": 0.2, "euclid": 0.1, "hst": 0.05, "hst_up": 0.03, "ao": 0.01}
pixel_scale = pixel_scales_dict[instrument]

dataset_path = Path("dataset") / "imaging" / "instruments" / instrument

dataset = aa.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
    over_sample_size_pixelization=4,
)

mask = aa.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)


"""
Reshape Dataset so that its exactly paired to the extent PSF convolution goes over including the blurring mask edge.

This speeds up JAX calculations as the PSF convolution is done on a smaller array with fewer zero entries.

This will be put in the source code soon during `apply_mask`.
"""


def false_span(mask: np.ndarray):
    """
    Given a boolean mask with False marking valid pixels,
    return the (y_min, y_max), (x_min, x_max) spans of False entries.
    """
    # Find coordinates of False pixels
    ys, xs = np.where(~mask)

    if ys.size == 0 or xs.size == 0:
        raise ValueError("No False entries in mask!")

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    return (y_max - y_min, x_max - x_min)


y_distance, x_distance = false_span(mask=mask.mask)

(pad_y, pad_x) = dataset.psf.shape_native

new_shape = (y_distance + pad_y, x_distance + pad_x)

mask = mask.resized_from(new_shape=new_shape)
data = dataset.data.resized_from(new_shape=new_shape)
noise_map = dataset.noise_map.resized_from(new_shape=new_shape)

dataset = aa.Imaging(
    data=data,
    noise_map=noise_map,
    psf=dataset.psf,
    over_sample_size_pixelization=4,
)

dataset = dataset.apply_mask(mask=mask)


"""
__Shapes__
"""
print(f"Data Shape: {dataset.data.shape_native}")
print(f"PSF Shape: {dataset.psf.shape_native}")
print(f"Mapping Matrix Shape: {mapping_matrix.shape}")
print("Curvature Matrix Shape: ", curvature_matrix.shape)


def blurred_mapping_matrix_from(psf, mapping_matrix):
    return jnp.hstack(
        [psf.convolve_mapping_matrix(mapping_matrix=mapping_matrix, mask=mask, jax_method="fft")]
    )


def curvature_matrix_via_mapping_matrix_from(
    mapping_matrix: np.ndarray,
    noise_map: np.ndarray,
) -> np.ndarray:

    array = mapping_matrix / noise_map[:, None]
    curvature_matrix = jnp.dot(array.T, array)

    return curvature_matrix


def curvature_matrix_via_w_tilde_from(
    w_tilde: np.ndarray, mapping_matrix: np.ndarray
) -> np.ndarray:
    return jnp.dot(mapping_matrix.T, jnp.dot(w_tilde, mapping_matrix))


jitted_blurred_mapping_matrix_from = jax.jit(blurred_mapping_matrix_from)
jitted_curvature_matrix_via_mapping_matrix_from = jax.jit(
    curvature_matrix_via_mapping_matrix_from
)
jitted_curvature_matrix_via_w_tilde_from = jax.jit(curvature_matrix_via_w_tilde_from)

"""
Precompute functions so compute tile not printed.
"""
from autoarray.inversion.inversion import inversion_util

mapping_matrix = jnp.array(mapping_matrix)

blurred_mapping_matrix_calc = jitted_blurred_mapping_matrix_from(
    psf=dataset.psf, mapping_matrix=mapping_matrix
)

curvature_matrix_calc = jitted_curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix_calc, noise_map=dataset.noise_map.array
)

curvature_matrix_w_tilde = jitted_curvature_matrix_via_w_tilde_from(
    w_tilde=w_matrix, mapping_matrix=mapping_matrix
)


"""
__Time JIT__
"""
start = time.time()

blurred_mapping_matrix_calc = jitted_blurred_mapping_matrix_from(
    psf=dataset.psf, mapping_matrix=mapping_matrix
)
print(blurred_mapping_matrix_calc[0, 0])

print(f"Time JAX jit mapping matrix: {time.time() - start}")

start = time.time()

curvature_matrix_calc = jitted_curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix_calc, noise_map=dataset.noise_map.array
)
print(curvature_matrix_calc[0, 0])


print(f"Time JAX jit curvature_matrix: {time.time() - start}")

start = time.time()

curvature_matrix_w_tilde = jitted_curvature_matrix_via_w_tilde_from(
    w_tilde=w_matrix, mapping_matrix=mapping_matrix
)
print(curvature_matrix_w_tilde[0, 0])

print(f"Time JAX jit curvature_matrix w_tilde: {time.time() - start}")


assert np.allclose(curvature_matrix_calc, curvature_matrix)


# batch_size = 10
#
# def blurred_mapping_matrix_from(psf, mapping_matrix):
#     mapping_matrix = jnp.hstack([psf.convolve_mapping_matrix(mapping_matrix=mapping_matrix, mask=mask)])
#     return 1
#
# def curvature_matrix_via_mapping_matrix_from(
#     mapping_matrix: np.ndarray,
#     noise_map: np.ndarray,
# ) -> np.ndarray:
#
#     array = mapping_matrix / noise_map[:, None]
#     curvature_matrix = jnp.dot(array.T, array)
#
#     return 1
#
#
# def curvature_matrix_via_w_tilde_from(
#     w_tilde: np.ndarray, mapping_matrix: np.ndarray
# ) -> np.ndarray:
#     curvature_matrix = jnp.dot(mapping_matrix.T, jnp.dot(w_tilde, mapping_matrix))
#     return 1
#
#
# vmap_blurred_mapping_matrix_from = jax.vmap(jax.jit(blurred_mapping_matrix_from))
# vmap_curvature_matrix_via_mapping_matrix_from = jax.vmap(jax.jit(curvature_matrix_via_mapping_matrix_from))
# vmap_curvature_matrix_via_w_tilde_from = jax.vmap(jax.jit(curvature_matrix_via_w_tilde_from))
#
# psf_list = batch_size([dataset.psf])
# mapping_matrix_list = batch_size([mapping_matrix])
#
# vmap_blurred_mapping_matrix_from


"""
__Time VMap__
"""
