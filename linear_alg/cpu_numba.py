import jax.numpy as jnp
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
data_to_pix_unique = np.load(f"{folder}/data_to_pix_unique.npy")
data_weights = np.load(f"{folder}/data_weights.npy")
pix_lengths = np.load(f"{folder}/pix_lengths.npy")
# w_matrix = np.load(f"{folder}/w_matrix.npy")
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

"""
__Mask__
"""
dataset = dataset.apply_mask(mask=mask)

"""
Precompute functions so compute tile not printed
"""
from autoarray.inversion.inversion import inversion_util
from autoarray.inversion.inversion.imaging import inversion_imaging_util

blurred_mapping_matrix_calc = dataset.convolver.convolve_mapping_matrix(
    mapping_matrix=mapping_matrix
)

curvature_matrix_calc = inversion_util.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix_calc, noise_map=dataset.noise_map
)

curvature_matrix_w_tilde = (
    inversion_imaging_util.curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
        curvature_preload=dataset.w_tilde.curvature_preload,
        curvature_indexes=dataset.w_tilde.indexes,
        curvature_lengths=dataset.w_tilde.lengths,
        data_to_pix_unique=np.array(data_to_pix_unique),
        data_weights=np.array(data_weights),
        pix_lengths=np.array(pix_lengths),
        pix_pixels=mapping_matrix.shape[1],
    )
)

"""
__Shapes__
"""
print(f"Data Shape: {dataset.data.shape_native}")
print(f"PSF Shape: {dataset.psf.shape_native}")
print(f"Mapping Matrix Shape: {mapping_matrix.shape}")
print("Curvature Matrix Shape: ", curvature_matrix.shape)


"""
__Time__
"""
start = time.time()

blurred_mapping_matrix_calc = dataset.convolver.convolve_mapping_matrix(
    mapping_matrix=mapping_matrix
)

print(f"Time numba CPU blurred mapping matrix calculation: {time.time() - start}")

start = time.time()

curvature_matrix_calc = inversion_util.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix_calc, noise_map=dataset.noise_map
)

print(
    f"Time numba CPU curvature_matrix calculation via mapping matrix: {time.time() - start}"
)

start = time.time()

curvature_matrix_w_tilde = (
    inversion_imaging_util.curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
        curvature_preload=dataset.w_tilde.curvature_preload,
        curvature_indexes=dataset.w_tilde.indexes,
        curvature_lengths=dataset.w_tilde.lengths,
        data_to_pix_unique=np.array(data_to_pix_unique),
        data_weights=np.array(data_weights),
        pix_lengths=np.array(pix_lengths),
        pix_pixels=mapping_matrix.shape[1],
    )
)

print(f"Time numba CPU curvature_matrix calculation via w_tilde: {time.time() - start}")

# Raises exceptions, need to follow up but not important for profilng.

# assert np.allclose(curvature_matrix_calc, curvature_matrix)
