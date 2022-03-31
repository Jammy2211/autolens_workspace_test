from astropy import units
import numpy as np
import os
from os import path

import autolens as al


def image_plane_grid(resolution, ratio=2.0, size=5.0):
    def power(x):

        return int(pow(2, np.ceil(np.log(x) / np.log(2.0))))

    print("The resolution is ~ {} arcsec".format(resolution))
    if ratio >= 2.0:
        pixel_scale_maximum = size / (resolution / ratio)
    else:
        raise ValueError("The pixel scale needs to be at least half the resolution.")

    n_pixels = power(x=pixel_scale_maximum)

    pixel_scale = size / n_pixels

    return n_pixels, pixel_scale


def image_plane_grid_from_uv_wavelengths(uv_wavelengths, size=5.0):

    uv_distance = np.hypot(uv_wavelengths[..., 0], uv_wavelengths[..., 1])

    resolution = 1.0 / np.max(uv_distance) * units.rad.to(units.arcsec)

    return image_plane_grid(resolution=resolution, size=size)


instrument = "sma"
# dataset_path = path.join("dataset", "interferometer", "instruments", "sma")
dataset_path = path.join("dataset", "interferometer", "instruments", "alma_high_res")

uv_wavelengths_path = path.join(dataset_path, "uv_wavelengths.fits")

uv_wavelengths = al.util.array_2d.numpy_array_2d_via_fits_from(
    file_path=uv_wavelengths_path, hdu=0
)

grid = image_plane_grid_from_uv_wavelengths(uv_wavelengths=uv_wavelengths)

print(grid)
