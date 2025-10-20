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

mapping_matrix = np.load(f"{folder}/mapping_matrix.npy")

# print(mapping_matrix.shape)
#
# for i in range(mapping_matrix.shape[1]):
#
#     print(i, np.count_nonzero(mapping_matrix[:,i]))
#
# fff

curvature_matrix = np.load(f"{folder}/curvature_matrix.npy")

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
        [
            psf.convolved_mapping_matrix_from(
                mapping_matrix=mapping_matrix, mask=mask, jax_method="direct"
            )
        ]
    )


def curvature_matrix_via_mapping_matrix_from(
    mapping_matrix: np.ndarray,
    noise_map: np.ndarray,
) -> np.ndarray:

    array = mapping_matrix / noise_map[:, None]
    curvature_matrix = jnp.dot(array.T, array)

    return curvature_matrix


jitted_blurred_mapping_matrix_from = jax.jit(blurred_mapping_matrix_from)
jitted_curvature_matrix_via_mapping_matrix_from = jax.jit(
    curvature_matrix_via_mapping_matrix_from
)

"""
Precompute functions so compute tile not printed.
"""
mapping_matrix = jnp.array(mapping_matrix)

blurred_mapping_matrix_calc = jitted_blurred_mapping_matrix_from(
    psf=dataset.psf, mapping_matrix=mapping_matrix
)

curvature_matrix_calc = jitted_curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix_calc, noise_map=dataset.noise_map.array
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


"""
__Time VMap__
"""
