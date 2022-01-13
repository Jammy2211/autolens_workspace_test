"""
__PROFILING: Inversion DelaunayMagnification__

This profiling script times how long it takes to fit `Imaging` data with a `DelaunayMagnification` pixelization for
datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoLens** log likelihood function.
"""
import numpy as np
import os
from os import path
from scipy.optimize import minimize_scalar
import sys

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "fit"))

import json

import autolens as al
import autolens.plot as aplt

"""
The path all profiling results are output.
"""

file_path = os.path.join(
    "imaging",
    "stochastic",
    "delaunay",
    "magnification",
    "constant",
    "samples_1d_static_source_pix__wrong_model",
)

"""
These settings control various aspects of how the fit is performed and therefore how stochasticity manifests.
"""
stochastic_seed = 1
sub_size = 4
mask_radius = 3.0
pixelization_shape_2d = (60, 60)

print(f"stochastic_seed = {stochastic_seed}")
print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"pixelization shape = {pixelization_shape_2d}")

"""
The lens galaxy used to fit the data, which is identical to the lens galaxy used to simulate the data. 
"""
lens_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=4.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp.EllExponential(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=2.0,
        effective_radius=1.6,
    ),
    mass=al.mp.EllPowerLaw(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=45.0),
        slope=2.0,
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.001, 0.001)),
)

lens_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=4.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp.EllExponential(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=2.0,
        effective_radius=1.6,
    ),
    mass=al.mp.EllPowerLaw(
        centre=(0.001, 0.001),
        einstein_radius=1.582,
        elliptical_comps=(0.091, 0.0),
        slope=1.9,
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.001, 0.001)),
)

"""
The source galaxy whose `DelaunayMagnification` `Pixelization` fits the data.
"""
pixelization = al.pix.DelaunayMagnification(shape=pixelization_shape_2d)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy_1 = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
    regularization=al.reg.Constant(coefficient=1.0),
)

"""
The simulated data comes at five resolution corresponding to five telescopes:
hst: pixel_scale = 0.05", normal run times, represents the type of data we do most our fitting on currently.
"""
instrument = "hst"

pixel_scale = 0.05

"""
Load the dataset for this instrument / resolution.
"""
dataset_path = path.join("dataset", "imaging", "instruments", instrument)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scale,
)

"""
Apply the 2D mask, which for the settings above is representative of the masks we typically use to model strong lenses.
"""
# mask = al.Mask2D.circular(
#     shape_native=imaging.shape_native,
#     pixel_scales=imaging.pixel_scales,
#     sub_size=sub_size,
#     radius=mask_radius,
# )

mask = al.Mask2D.circular_annular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    sub_size=sub_size,
    inner_radius=1.0,
    outer_radius=3.0,
)

masked_imaging = imaging.apply_mask(mask=mask)

masked_imaging = masked_imaging.apply_settings(
    settings=al.SettingsImaging(sub_size=sub_size)
)


"""
__Coefficient__

Determine the regularization coefficient using the correct lens model.
"""

def func_0(coefficient):

    source_galaxy_0.regularization.coefficient = coefficient
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy_0, source_galaxy_0])
    fit = al.FitImaging(
        dataset=masked_imaging,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(use_border=False),
    )

    fom = fit.figure_of_merit

    print(coefficient, fom)

    return -1.0 * fom


print("\nSetting Regularization Coefficient\n")

#coefficient_0 = minimize_scalar(func_0, method="bounded", bounds=[1e-3, 1e3]).x
coefficient_0 = 1.3191267732710674
print(f"coefficient_0 = {coefficient_0}")

def func_1(coefficient):

    source_galaxy_1.regularization.coefficient = coefficient
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy_1, source_galaxy_1])
    fit = al.FitImaging(
        dataset=masked_imaging,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(use_border=False),
    )

    fom = fit.figure_of_merit

    print(coefficient, fom)

    return -1.0 * fom


print("\nSetting Regularization Coefficient\n")

#coefficient_1 = minimize_scalar(func_1, method="bounded", bounds=[1e-3, 1e3]).x

coefficient_1 = 1.306669217584403
print(f"coefficient_1 = {coefficient_1}")

source_galaxy_1.regularization.coefficient = coefficient_1

"""
Output an image of the fit, so that we can inspect that it fits the data as expected.
"""
tracer_0 = al.Tracer.from_galaxies(galaxies=[lens_galaxy_0, source_galaxy_0])

fit = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer_0,
    settings_pixelization=al.SettingsPixelization(use_border=False),
)

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_fit_imaging_0", format="png"
    )
)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_imaging_plotter.subplot_fit_imaging()

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_of_plane_1_0", format="png"
    )
)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_imaging_plotter.subplot_of_planes(plane_index=1)


tracer_1 = al.Tracer.from_galaxies(galaxies=[lens_galaxy_1, source_galaxy_1])

fit = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer_1,
    settings_pixelization=al.SettingsPixelization(use_border=False),
)

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_fit_imaging_1", format="png"
    )
)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_imaging_plotter.subplot_fit_imaging()

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_of_plane_1_1", format="png"
    )
)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_imaging_plotter.subplot_of_planes(plane_index=1)

"""
Preload fixed grid
"""

traced_sparse_grids_list_of_planes, sparse_image_plane_grid_list = tracer_0.traced_sparse_grid_pg_list_from(
    grid=masked_imaging.grid_inversion,
    settings_pixelization=al.SettingsPixelization(use_border=False),
)

preloads_0 = al.Preloads(
    static_pixelization=traced_sparse_grids_list_of_planes[1][0]
)

traced_sparse_grids_list_of_planes, sparse_image_plane_grid_list = tracer_1.traced_sparse_grid_pg_list_from(
    grid=masked_imaging.grid_inversion,
    settings_pixelization=al.SettingsPixelization(use_border=False),
)

preloads_1 = al.Preloads(
    static_pixelization=traced_sparse_grids_list_of_planes[1][0]
)


tracer_0.galaxies[1].pixelization = al.pix.DelaunayMagnificationStatic(shape=pixelization_shape_2d)
tracer_1.galaxies[1].pixelization = al.pix.DelaunayMagnificationStatic(shape=pixelization_shape_2d)


"""
__Stochastic Calculation__

Create new Tracer with variable density slope and fit for each lens model, storing all likelihood terms.
"""
fit = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer_0,
    preloads=preloads_0,
    settings_pixelization=al.SettingsPixelization(use_border=False),
)

print(f"True model w/ True model source pixelization: {fit.figure_of_merit}")

fit = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer_1,
    preloads=preloads_1,
    settings_pixelization=al.SettingsPixelization(use_border=False),
)

print(f"Wrong model w/ Wrong model source pixelization: {fit.figure_of_merit}")

fit = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer_0,
    preloads=preloads_1,
    settings_pixelization=al.SettingsPixelization(use_border=False),
)

print(f"True model w/ Wrong model source pixelization: {fit.figure_of_merit}")

fit = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer_1,
    preloads=preloads_0,
    settings_pixelization=al.SettingsPixelization(use_border=False),
)

print(f"Wrong model w/ True model source pixelization: {fit.figure_of_merit}")