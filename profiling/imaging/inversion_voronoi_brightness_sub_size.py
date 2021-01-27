import time
from os import path
import numpy as np

import autolens as al
import autolens.plot as aplt

repeats = 1

print("Number of repeats = " + str(repeats))
print()

sub_size = 2
radius = 3.5
psf_shape_2d = (11, 11)
pixels = 1000

print("sub grid size = " + str(sub_size))
print("circular mask radius = " + str(radius) + "\n")
print("psf shape = " + str(psf_shape_2d) + "\n")
print("pixels = " + str(pixels) + "\n")

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=45.0),
    ),
)

pixelization = al.pix.VoronoiBrightnessImage(pixels=pixels, weight_power=10.0)

# pixelization = al.pix.VoronoiMagnification(shape=(40, 40))

instruments = ["hst"]  # , "hst_up"]
pixel_scales = [0.05]  # , 0.03]

sub_sizes = list(range(1, 16))

for instrument, pixel_scale in zip(instruments, pixel_scales):

    log_evidences = []

    for sub_size in sub_sizes:

        dataset_path = path.join("dataset", "instruments", instrument)

        imaging = al.Imaging.from_fits(
            image_path=path.join(dataset_path, "image.fits"),
            psf_path=path.join(dataset_path, "psf.fits"),
            noise_map_path=path.join(dataset_path, "noise_map.fits"),
            pixel_scales=pixel_scale,
        )

        mask = al.Mask2D.circular(
            shape_native=imaging.shape_native,
            pixel_scales=imaging.pixel_scales,
            sub_size=sub_size,
            radius=radius,
        )

        grid = al.Grid2D.from_mask(mask=mask.mask_sub_1)

        masked_imaging = al.MaskedImaging(
            imaging=imaging,
            mask=mask,
            settings=al.SettingsMaskedImaging(sub_size=sub_size),
        )

        source_galaxy = al.Galaxy(
            redshift=1.0,
            pixelization=pixelization,
            regularization=al.reg.Constant(coefficient=1.0),
            hyper_model_image=masked_imaging.image,
            hyper_galaxy_image=masked_imaging.image,
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        sparse_grid = pixelization.sparse_grid_from_grid(
            grid=grid, hyper_image=masked_imaging.image
        )

        sparse_grids_of_planes = tracer.sparse_image_plane_grids_of_planes_from_grid(
            grid=grid
        )

        settings_pixelization = al.SettingsPixelization(
            preload_sparse_grids_of_planes=sparse_grids_of_planes
        )

        fit = al.FitImaging(
            masked_imaging=masked_imaging,
            tracer=tracer,
            settings_pixelization=settings_pixelization,
        )

        print(sub_size, fit.log_evidence, str(masked_imaging.grid.sub_shape_slim))

        log_evidences.append(fit.log_evidence)
        #
        # fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)
        # fit_imaging_plotter.subplot_fit_imaging()
        # fit_imaging_plotter.subplot_of_planes(plane_index=1)
        #
        # stop

import matplotlib.pyplot as plt

plt.plot(sub_sizes, log_evidences)
plt.ylabel("Evidence")
plt.xlabel("Sub size")
plt.show()
