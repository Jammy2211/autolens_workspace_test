"""
__PROFILING: Inversion Voronoi__

This profiling script times how long it takes to fit `Imaging` data with a `Voronoi` pixelization for
datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoLens** log likelihood function.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

from scipy.optimize import minimize_scalar

import autolens as al
import autolens.plot as aplt


"""
The path all profiling results are output.
"""
file_path = os.path.join(
    "imaging",
    "profiling",
    "inversion",
    "voronoi_nn",
    "magnification",
    "constant",
    "noise_not_split",
    al.__version__,
)

"""
Whether w_tilde is used dictates the output folder.
"""
use_w_tilde = True

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 4
mask_radius = 3.5
psf_shape_2d = (21, 21)
mesh_shape_2d = (60, 60)


print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"psf shape = {psf_shape_2d}")
print(f"pixelization shape = {mesh_shape_2d}")

"""
The lens galaxy used to fit the data, which is identical to the lens galaxy used to simulate the data. 
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.001, gamma_2=0.001),
)

"""
The source galaxy whose `Voronoi` `Pixelization` fits the data.
"""
image_mesh = al.image_mesh.Overlay(shape=mesh_shape_2d)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.Pixelization(
        image_mesh=image_mesh,
        mesh=al.mesh.Voronoi(),
        regularization=al.reg.Constant(coefficient=1.0),
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
The simulated data comes at five resolution corresponding to five telescopes:

vro: pixel_scale = 0.2", fastest run times.
euclid: pixel_scale = 0.1", fast run times
hst: pixel_scale = 0.05", normal run times, represents the type of data we do most our fitting on currently.
hst_up: pixel_scale = 0.03", slow run times.
ao: pixel_scale = 0.01", very slow :(
"""
instrument = "hst_noise"
pixel_scale = 0.05

"""
Load the dataset for this instrument / resolution.
"""
dataset_path = path.join("dataset", "imaging", "instruments", instrument)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scale,
)

"""
Apply the 2D mask, which for the settings above is representative of the masks we typically use to model strong lenses.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    sub_size=sub_size,
    radius=mask_radius,
)

masked_dataset = dataset.apply_mask(mask=mask)

masked_dataset = masked_dataset.apply_settings(
    settings=al.SettingsImaging(sub_size=sub_size)
)


def func(coefficient):
    source_galaxy.pixelization.regularization.coefficient = coefficient
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    fit = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_border_relocator=True),
    )

    fom = fit.figure_of_merit

    print(coefficient, fom)

    return -1.0 * fom


print("\nSetting Regularization Coefficient\n")

coefficient = minimize_scalar(func, method="bounded", bounds=[1e-3, 1e3]).x

print(f"coefficient = {coefficient}")

source_galaxy.pixelization.regularization.coefficient = coefficient

"""
__Fit Time__

Time FitImaging by itself, to compare to profiling dict call.
"""
fit = al.FitImaging(
    dataset=masked_dataset,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(use_w_tilde=use_w_tilde),
)


"""
Output an image of the fit, so that we can inspect that it fits the data as expected.
"""
mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_fit", format="png"
    )
)
fit_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_fit()

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_of_plane_1", format="png"
    )
)
fit_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_of_planes(plane_index=1)
