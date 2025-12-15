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

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=mask_radius
)

def make_mask_rectangular(mask, dataset):
    ys, xs = np.where(~mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    (pad_y, pad_x) = dataset.psf.shape_native
    z = np.ones(mask.shape, dtype=bool)
    shape = ((y_max + pad_y // 2) - (y_min - pad_y // 2), (x_max + pad_x // 2) - (x_min - pad_x // 2))
    return z, shape

new_mask_array, mask_shape = make_mask_rectangular(mask, dataset)

# mask._array = new_mask_array

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
    mesh=al.mesh.RectangularMagnification(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])


image_2d = tracer.image_2d_from(grid=dataset.grids.lp)
blurring_image_2d = tracer.image_2d_from(grid=dataset.grids.blurring)

import scipy

psf_native = jnp.array(dataset.psf.native.array)

full_shape = tuple(s1 + s2 - 1 for s1, s2 in zip(mask_shape, psf_native.shape))
fft_shape = tuple(scipy.fft.next_fast_len(s, real=True) for s in full_shape)

fft_psf = jnp.fft.rfft2(psf_native, s=fft_shape)

def blurred_image_from(fft_psf, image, blurring_image):
    
    slim_to_native_tuple = dataset.psf.slim_to_native_tuple
    slim_to_native_blurring_tuple = dataset.psf.slim_to_native_blurring_tuple

    # make sure dtype matches what you want
    image_both_native = jnp.zeros(
        mask.shape, dtype=image.dtype
    )

    # set using a tuple of index arrays
    image_both_native = image_both_native.at[slim_to_native_tuple].set(
        jnp.asarray(image)
    )
    image_both_native = image_both_native.at[
        slim_to_native_blurring_tuple
    ].set(jnp.asarray(blurring_image))

    # FFT the combined image
    fft_image_native = jnp.fft.rfft2(image_both_native, s=fft_shape, axes=(0, 1))

    # Multiply by PSF in Fourier space and invert
    blurred_image_full = jnp.fft.irfft2(fft_psf * fft_image_native, s=fft_shape, axes=(0, 1))

    # Crop back to mask_shape
    start_indices = tuple((full_size - out_size) // 2 for full_size, out_size in zip(full_shape, mask_shape))
    out_shape_full = mask_shape
    blurred_image_native = jax.lax.dynamic_slice(blurred_image_full, start_indices, out_shape_full)

    # Return only unmasked pixels (slim form)
    return blurred_image_native[slim_to_native_tuple]

import time

start = time.time()

blurred_image_from_jit = jax.jit(blurred_image_from)

print(f"JIT FFT compile time: {time.time() - start}")

start = time.time()

blurred_image_2d_via_fft = blurred_image_from_jit(
    fft_psf, jnp.array(image_2d.array), jnp.array(blurring_image_2d.array)
)


blurred_image_2d_via_fft = al.Array2D(
    values=blurred_image_2d_via_fft, mask=image_2d.mask
)
print(blurred_image_2d_via_fft[0])
print(f"JIT FFT run time: {time.time() - start}")



def convolved_image_from(image, blurring_image, jax_method="direct"):
    """
    For a given 1D array and blurring array, convolve the two using this psf.

    Parameters
    ----------
    image
        1D array of the values which are to be blurred with the psf's PSF.
    blurring_image
        1D array of the blurring values which blur into the array after PSF convolution.
    jax_method
        If JAX is enabled this keyword will indicate what method is used for the PSF
        convolution. Can be either `direct` to calculate it in real space or `fft`
        to calculated it via a fast Fourier transform. `fft` is typically faster for
        kernels that are more than about 5x5. Default is `fft`.
    """

    slim_to_native_tuple = dataset.psf.slim_to_native_tuple
    slim_to_native_blurring_tuple = dataset.psf.slim_to_native_blurring_tuple

    # make sure dtype matches what you want
    expanded_array_native = jnp.zeros(
        image.mask.shape, dtype=jnp.asarray(image.array).dtype
    )

    # set using a tuple of index arrays
    expanded_array_native = expanded_array_native.at[slim_to_native_tuple].set(
        jnp.asarray(image.array)
    )
    expanded_array_native = expanded_array_native.at[
        slim_to_native_blurring_tuple
    ].set(jnp.asarray(blurring_image.array))

    kernel = dataset.psf.stored_native.array

    convolve_native = jax.scipy.signal.convolve(
        expanded_array_native, kernel, mode="same", method=jax_method
    )

    convolved_array_1d = convolve_native[slim_to_native_tuple]

    return convolved_array_1d


start = time.time()

convolve_image_jit = jax.jit(convolved_image_from)
print(f"JIT compile time real space: {time.time() - start}")

start = time.time()

blurred_image_2d_via_real_space = convolve_image_jit(
    image=image_2d,
    blurring_image=blurring_image_2d,
)
blurred_image_2d_via_real_space = al.Array2D(values=blurred_image_2d_via_real_space, mask=dataset.mask)
print(blurred_image_2d_via_real_space[0])
print(f"JIT run time real space: {time.time() - start}")

print(blurred_image_2d_via_fft.native.shape)
print(blurred_image_2d_via_real_space.native.shape)

residuals = (
    blurred_image_2d_via_fft.native.array - blurred_image_2d_via_real_space.native.array
)

import matplotlib.pyplot as plt

plt.imshow(residuals)
plt.savefig("residuals.png")


print(f"Max diff {np.max(blurred_image_2d_via_fft - blurred_image_2d_via_real_space)}")





















