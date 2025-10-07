import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import autolens as al

# instrument = "vro"
# instrument = "euclid"
instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

pixel_scales_dict = {"vro": 0.2, "euclid": 0.1, "hst": 0.05, "hst_up": 0.03, "ao": 0.01}
pixel_scale = pixel_scales_dict[instrument]

mesh_shape = (32, 32)

dataset_path = Path("dataset") / "imaging" / "instruments" / instrument

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
    over_sample_size_pixelization=4,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

"""
Reshape Dataset so that its exactly paired to the extent PSF convolution goes over including the blurring mask edge.

This speeds up JAX calculations as the PSF convolution is done on a smaller array with fewer zero entries.

This will be put in the source code soon during `apply_mask`.
"""
# def false_span(mask: np.ndarray):
#     """
#     Given a boolean mask with False marking valid pixels,
#     return the (y_min, y_max), (x_min, x_max) spans of False entries.
#     """
#     # Find coordinates of False pixels
#     ys, xs = np.where(~mask)
#
#     if ys.size == 0 or xs.size == 0:
#         raise ValueError("No False entries in mask!")
#
#     y_min, y_max = ys.min(), ys.max()
#     x_min, x_max = xs.min(), xs.max()
#
#     return (y_max - y_min, x_max - x_min)
#
#
# y_distance, x_distance = false_span(mask=mask.mask)
#
# (pad_y, pad_x) = dataset.psf.shape_native
#
# new_shape = (y_distance + pad_y, x_distance + pad_x)
#
# mask = mask.resized_from(new_shape=new_shape)
# data = dataset.data.resized_from(new_shape=new_shape)
# noise_map = dataset.noise_map.resized_from(new_shape=new_shape)
#
# dataset = al.Imaging(
#     data=data,
#     noise_map=noise_map,
#     psf=dataset.psf,
#     over_sample_size_pixelization=4,
# )
#

dataset = dataset.apply_mask(mask=mask)


lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=4.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=2.0,
        effective_radius=1.6,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.001, gamma_2=0.001),
)

pixelization = al.Pixelization(
    mesh=al.mesh.Rectangular(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

import scipy

# Data shape is (180, 180), this is based on the .fits file and currently does not change or resize.
# The FFT size is (160, 160) which is the next fast length for FFTs greater than (180+15, 180+15) where (15, 15) is the PSF shape.
# The mask_shape is (139, 139) which is the rectangular region enclosing the non-masked data.

# Is the right code implementation to internally resize the data and noise map to the next fast length for FFTs?


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
psf_native = jnp.array(dataset.psf.native.array)

full_shape = tuple(s1 + s2 - 1 for s1, s2 in zip(mask_shape, psf_native.shape))
fft_shape = tuple(scipy.fft.next_fast_len(s, real=True) for s in full_shape)

# ALSO CHECK RULES FOR CENTERING

fft_shape = (180, 180)

fft_psf = jnp.fft.rfft2(psf_native, s=fft_shape)

mask = mask.resized_from(new_shape=fft_shape, pad_value=True)
data = dataset.data.resized_from(new_shape=fft_shape)
noise_map = dataset.noise_map.resized_from(new_shape=fft_shape)

dataset = al.Imaging(
    data=data,
    noise_map=noise_map,
    psf=dataset.psf,
    over_sample_size_pixelization=4,
    check_noise_map=False,
)

image_2d = tracer.image_2d_from(grid=dataset.grids.lp)
blurring_image_2d = tracer.image_2d_from(grid=dataset.grids.blurring)


def blurred_image_from(fft_psf, image, blurring_image):

    image_native = image + blurring_image

    fft_image_native = jnp.fft.rfft2(image_native, s=fft_shape, axes=(0, 1))
    blurred_image_full = jnp.fft.irfft2(
        fft_psf * fft_image_native, s=fft_shape, axes=(0, 1)
    )

    return blurred_image_full

    # out_shape = full_shape
    # start_indices = tuple(
    #     (full_size - out_size) // 2
    #     for full_size, out_size in zip(blurred_image_full.shape, out_shape)
    # )
    # blurred_image_native = jax.lax.dynamic_slice(blurred_image_full, start_indices, out_shape)
    # return blurred_image_native


blurred_image_2d_via_fft = blurred_image_from(
    fft_psf, jnp.array(image_2d.native.array), jnp.array(blurring_image_2d.native.array)
)


blurred_image_2d_via_fft = al.Array2D(
    values=blurred_image_2d_via_fft, mask=image_2d.mask
)


image_2d = tracer.image_2d_from(grid=dataset.grids.lp)
blurring_image_2d = tracer.image_2d_from(grid=dataset.grids.blurring)

blurred_image_2d_via_real_space = dataset.psf.convolve_image_via_real_space(
    image=image_2d,
    blurring_image=blurring_image_2d,
)

print(blurred_image_2d_via_real_space.shape)
print(blurred_image_2d_via_fft.shape)

residuals = (
    blurred_image_2d_via_fft.native.array - blurred_image_2d_via_real_space.native.array
)

import matplotlib.pyplot as plt

plt.imshow(residuals)
plt.savefig("residuals.png")


print(f"Max diff {np.max(blurred_image_2d_via_fft - blurred_image_2d_via_real_space)}")
