import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import scipy

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

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=mask_radius
)

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

psf_native = jnp.array(dataset.psf.native.array)

full_shape = tuple(s1 + s2 - 1 for s1, s2 in zip(dataset.mask.shape_native, psf_native.shape))
# fft_shape = tuple(scipy.fft.next_fast_len(s, real=True) for s in full_shape)

fft_shape = dataset.data.shape_native

from jax.numpy.fft import ifftshift

psf_native = jnp.array(dataset.psf.native.array)
psf_native /= psf_native.sum()
psf_shifted = ifftshift(psf_native)
fft_psf = jnp.fft.rfft2(psf_shifted, s=fft_shape)

image_2d = tracer.image_2d_from(grid=dataset.grids.lp)
blurring_image_2d = tracer.image_2d_from(grid=dataset.grids.blurring)

def blurred_image_from(psf, image, blurring_image):

    Ny, Nx = image.shape
    Py, Px = psf.shape
    fft_shape = (Ny + Py - 1, Nx + Px - 1)

    # Normalize and shift PSF
    psf_native = jnp.array(dataset.psf.native.array, dtype=jnp.float32)  # ensure real
    psf_native /= psf_native.sum()  # normalize
    psf_shifted = ifftshift(psf_native)  # shift center→corner

    fft_psf = jnp.fft.rfft2(psf_shifted, s=fft_shape)

    # FFT of padded image
    image_native = image + blurring_image
    fft_image_native = jnp.fft.rfft2(image_native, s=fft_shape)

    # Convolution in Fourier space
    blurred_full = jnp.fft.irfft2(fft_psf * fft_image_native, s=fft_shape)

    # Crop back to original image size
    y0 = (fft_shape[0] - Ny) // 2
    x0 = (fft_shape[1] - Nx) // 2
    return blurred_full[y0:y0+Ny, x0:x0+Nx]

blurred_image_2d_via_fft = blurred_image_from(
    fft_psf, jnp.array(image_2d.native.array), jnp.array(blurring_image_2d.native.array)
)

blurred_image_2d_via_fft = al.Array2D(
    values=blurred_image_2d_via_fft, mask=image_2d.mask
)


image_2d = tracer.image_2d_from(grid=dataset.grids.lp)
blurring_image_2d = tracer.image_2d_from(grid=dataset.grids.blurring)

blurred_image_2d_via_real_space = dataset.psf.convolved_image_via_real_space_from(
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
plt.close()

plt.imshow(blurred_image_2d_via_fft.native.array)
plt.savefig("fft.png")
plt.close()

plt.imshow(blurred_image_2d_via_real_space.native.array)
plt.savefig("real_space.png")
plt.close()


# print 2d argmax of images
print(np.unravel_index(np.argmax(blurred_image_2d_via_fft.native), blurred_image_2d_via_fft.shape_native))
print(np.unravel_index(np.argmax(blurred_image_2d_via_real_space.native), blurred_image_2d_via_real_space.shape_native))

print(f"Max diff {np.max(blurred_image_2d_via_fft - blurred_image_2d_via_real_space)}")
