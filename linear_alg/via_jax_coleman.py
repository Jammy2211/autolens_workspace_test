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
    z[
        y_min - pad_y // 2 : y_max + pad_y // 2, x_min - pad_x // 2 : x_max + pad_x // 2
    ] = False
    shape = (
        (y_max + pad_y // 2) - (y_min - pad_y // 2),
        (x_max + pad_x // 2) - (x_min - pad_x // 2),
    )
    return z, shape


new_mask_array, mask_shape = make_mask_rectangular(mask, dataset)
mask._array = new_mask_array

# """
# Reshape Dataset so that its exactly paired to the extent PSF convolution goes over including the blurring mask edge.

# This speeds up JAX calculations as the PSF convolution is done on a smaller array with fewer zero entries.

# This will be put in the source code soon during `apply_mask`.
# """
# def false_span(mask: np.ndarray):
#     """
#     Given a boolean mask with False marking valid pixels,
#     return the (y_min, y_max), (x_min, x_max) spans of False entries.
#     """
#     # Find coordinates of False pixels
#     ys, xs = np.where(~mask)

#     if ys.size == 0 or xs.size == 0:
#         raise ValueError("No False entries in mask!")

#     y_min, y_max = ys.min(), ys.max()
#     x_min, x_max = xs.min(), xs.max()

#     return (y_max - y_min, x_max - x_min)


# y_distance, x_distance = false_span(mask=mask.mask)

# (pad_y, pad_x) = dataset.psf.shape_native

# new_shape = (y_distance + pad_y, x_distance + pad_x)

# mask = mask.resized_from(new_shape=new_shape)
# data = dataset.data.resized_from(new_shape=new_shape)
# noise_map = dataset.noise_map.resized_from(new_shape=new_shape)

# dataset = aa.Imaging(
#     data=data,
#     noise_map=noise_map,
#     psf=dataset.psf,
#     over_sample_size_pixelization=4,
# )

dataset = dataset.apply_mask(mask=mask)

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
    mapping_matrix_native = mapping_matrix.reshape(mask_shape + (-1,))
    fft_mapping_matrix_native = jnp.fft.rfft2(
        mapping_matrix_native, s=fft_shape, axes=(0, 1)
    )
    blurred_mapping_matrix_full = jnp.fft.irfft2(
        fft_psf * fft_mapping_matrix_native, s=fft_shape, axes=(0, 1)
    )
    start_indices = tuple(
        (full_size - out_size) // 2
        for full_size, out_size in zip(full_shape, mask_shape)
    ) + (0,)
    out_shape_full = mask_shape + (blurred_mapping_matrix_full.shape[2],)
    blurred_mapping_matrix_native = jax.lax.dynamic_slice(
        blurred_mapping_matrix_full, start_indices, out_shape_full
    )
    return blurred_mapping_matrix_native.reshape(
        -1, blurred_mapping_matrix_native.shape[-1]
    )


def curvature_matrix_via_mapping_matrix_from(mapping_matrix, noise_map):

    # scale rows by 1/noise
    W = 1.0 / noise_map
    mapping_matrix_scaled = mapping_matrix * W[:, None]  # broadcast scaling

    # curvature = A^T A (with scaling included)
    curvature = mapping_matrix_scaled.T @ mapping_matrix_scaled

    # The result is dense (n_sources x n_sources)
    return curvature


# test around doing curvature calculation in FFT space
# mapping_matrix_native = (mapping_matrix / jnp.expand_dims(noise_jax, 1)).reshape(mask_shape + (-1,))
# fft_psf = jnp.fft.fft2(psf_native, s=fft_shape, norm='ortho')
# fft_psf = jnp.expand_dims(fft_psf, 2)
# fft_mapping_matrix_native = jnp.fft.fft2(mapping_matrix_native, s=fft_shape, axes=(0, 1), norm='ortho')
# fft_bmm = fft_psf * fft_mapping_matrix_native
# fft_bmm_flt = fft_bmm.reshape(-1, 1024)
# tst = (fft_bmm_flt.conj().T @ fft_bmm_flt).real
# tst.round(5)


start = time.time()

jitted_blurred_mapping_matrix_from = jax.jit(blurred_mapping_matrix_from)
jitted_curvature_matrix_via_mapping_matrix_from = jax.jit(
    curvature_matrix_via_mapping_matrix_from
)

print(f"Time JAX jit functions: {time.time() - start}")

"""
Precompute functions so compute tile not printed.
"""
blurred_mapping_matrix_calc = jitted_blurred_mapping_matrix_from(
    fft_psf, mapping_matrix
)

curvature_matrix_calc = jitted_curvature_matrix_via_mapping_matrix_from(
    blurred_mapping_matrix_calc, noise_jax
)

"""
__Time JIT__
"""
start = time.time()

blurred_mapping_matrix_calc = jitted_blurred_mapping_matrix_from(
    fft_psf, mapping_matrix
)

print(f"Time JAX mapping matrix run times: {time.time() - start}")

start = time.time()

curvature_matrix_calc = jitted_curvature_matrix_via_mapping_matrix_from(
    blurred_mapping_matrix_calc, noise_jax
)

print(f"Time JAX curvature matrix run times: {time.time() - start}")

print(np.max(curvature_matrix_calc))
print(np.min(curvature_matrix_calc))

"""
__Time VMap__
"""
