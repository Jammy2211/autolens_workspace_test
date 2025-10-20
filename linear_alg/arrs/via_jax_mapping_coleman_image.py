import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
from pathlib import Path
import numpy as np
import time

import autoarray as aa
import scipy

# instrument = "vro"
# instrument = "euclid"
instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

# folder = Path("linear_alg") / "arrs" / instrument
folder = Path("linear_alg") / "arrs" / instrument

mapping_matrix = np.load(f"{folder}/mapping_matrix.npy")
blurred_mapping_matrix_orig = np.load(f"{folder}/blurred_mapping_matrix.npy")
curvature_matrix = np.load(f"{folder}/curvature_matrix.npy")

pixel_scales_dict = {"vro": 0.2, "euclid": 0.1, "hst": 0.05, "hst_up": 0.03, "ao": 0.01}
pixel_scale = pixel_scales_dict[instrument]

# dataset_path = Path("dataset") / "imaging" / "instruments" / instrument
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

def make_mask_rectangular(mask, dataset):
    ys, xs = np.where(~mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    (pad_y, pad_x) = dataset.psf.shape_native
    z = np.ones(mask.shape, dtype=bool)
#     z[y_min - pad_y // 2:y_max + pad_y // 2, x_min - pad_x // 2:x_max + pad_x // 2] = False
    shape = ((y_max + pad_y // 2) - (y_min - pad_y // 2), (x_max + pad_x // 2) - (x_min - pad_x // 2))
    return z, shape


print(mask.shape)

new_mask_array, mask_shape = make_mask_rectangular(mask, dataset)

print(new_mask_array.shape)
print(mask_shape)

# mask._array = new_mask_array

dataset = dataset.apply_mask(mask=mask)

mask_shape = (180, 180)

"""
__Shapes__
"""
print(f"Data Shape: {dataset.data.shape_native}")
print(f"PSF Shape: {dataset.psf.shape_native}")
print(f"Mapping Matrix Shape: {mapping_matrix.shape}")
print("Curvature Matrix Shape: ", curvature_matrix.shape)

mapping_matrix = jnp.array(mapping_matrix)
psf_native = jnp.array(dataset.psf.native.array)
noise_jax = jnp.array(dataset.noise_map.array)

full_shape = tuple(s1 + s2 - 1 for s1, s2 in zip(mask_shape, psf_native.shape))
fft_shape = tuple(scipy.fft.next_fast_len(s, real=True) for s in full_shape)
fft_psf = jnp.fft.rfft2(psf_native, s=fft_shape)
fft_psf = jnp.expand_dims(fft_psf, 2)


def blurred_mapping_matrix_from(fft_psf, mapping_matrix):

    slim_to_native_tuple = dataset.psf.slim_to_native_tuple  # (y_idx, x_idx)
    n_src = mapping_matrix.shape[1]  # number of source pixels

    # start with zeros in native 2D shape + source dimension
    mapping_matrix_native = jnp.zeros(mask.shape + (n_src,), dtype=mapping_matrix.dtype)

    # scatter all source columns at once
    mapping_matrix_native = mapping_matrix_native.at[slim_to_native_tuple].set(mapping_matrix)

    fft_mapping_matrix_native = jnp.fft.rfft2(mapping_matrix_native, s=fft_shape, axes=(0, 1))
    blurred_mapping_matrix_full = jnp.fft.irfft2(fft_psf * fft_mapping_matrix_native, s=fft_shape, axes=(0, 1))
    start_indices = tuple((full_size - out_size) // 2 for full_size, out_size in zip(full_shape, mask_shape)) + (0,)
    out_shape_full = mask_shape + (blurred_mapping_matrix_full.shape[2],)
    blurred_mapping_matrix_native = jax.lax.dynamic_slice(blurred_mapping_matrix_full, start_indices, out_shape_full)

    return blurred_mapping_matrix_native[slim_to_native_tuple]


def curvature_matrix_via_mapping_matrix_from(mapping_matrix, noise_map):
    array = mapping_matrix / jnp.expand_dims(noise_map, 1)
    return jnp.dot(array.T, array)

start = time.time()

jitted_blurred_mapping_matrix_from = jax.jit(blurred_mapping_matrix_from)
jitted_curvature_matrix_via_mapping_matrix_from = jax.jit(curvature_matrix_via_mapping_matrix_from)

print(f"Time JAX jit functions: {time.time() - start}")

"""
Precompute functions so compute tile not printed.
"""
blurred_mapping_matrix_calc = jitted_blurred_mapping_matrix_from(fft_psf, mapping_matrix)

curvature_matrix_calc = jitted_curvature_matrix_via_mapping_matrix_from(blurred_mapping_matrix_calc, noise_jax)

"""
__Time JIT__
"""
start = time.time()

blurred_mapping_matrix_calc = jitted_blurred_mapping_matrix_from(fft_psf, mapping_matrix)

print(blurred_mapping_matrix_calc[0, 0])
print(f"Time JAx blurred time: {time.time() - start}")

start = time.time()

curvature_matrix_calc = jitted_curvature_matrix_via_mapping_matrix_from(blurred_mapping_matrix_calc, noise_jax)
print(curvature_matrix_calc[0, 0])

print(f"Time JAX curvature time: {time.time() - start}")

print(np.max(curvature_matrix_calc - curvature_matrix))

"""
__Time VMap__
"""
