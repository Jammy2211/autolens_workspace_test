"""
__PROFILING: Inversion VoronoiNNMagnification__

This profiling script times how long it takes to fit `Imaging` data with a `VoronoiNNMagnification` pixelization for
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


on_cosma = bool(int(sys.argv[1]))

"""
The path all profiling results are output.
"""
cosma_path = path.join(
    path.sep, "home", "dc-nigh1", "rds", "rds-dirac-dp195-i2FIP1t5TkY", "dc-nigh1"
)

if on_cosma:

    file_path = path.join(
        cosma_path,
        "output",
        "stochastic",
        "voronoi",
        "magnification",
        "constant",
        "samples_1d",
    )

else:
    file_path = os.path.join(path.relpath(path.dirname(__file__)), "samples_1d")

"""
This script varies the slope of the lens model in 1D to create a 1D plot showing the stochasticity. These settings 
control the range of slope values and interval over which the slope is varied.
"""
slope_lower = 1.99
slope_upper = 2.01
slope_interval = 0.00004

"""
These settings control various aspects of how the fit is performed and therefore how stochasticity manifests.
"""
stochastic_seed = 1
sub_size = 4
mask_radius = 3.0
mesh_shape_2d = (50, 50)

print(f"stochastic_seed = {stochastic_seed}")
print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"pixelization shape = {mesh_shape_2d}")

"""
The lens galaxy used to fit the data, which is identical to the lens galaxy used to simulate the data. 
"""
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
    mass=al.mp.PowerLaw(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        slope=2.0,
    ),
    shear=al.mp.ExternalShear(gamma_1=0.001, gamma_2=0.001),
)

source_galaxy_true = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

"""
The source galaxy whose `VoronoiNNMagnification` `Pixelization` fits the data.
"""
mesh = al.mesh.VoronoiNNMagnification(shape=mesh_shape_2d)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.Pixelization(
        mesh=mesh, regularization=al.reg.Constant(coefficient=1.0)
    ),
)

"""
The simulated data comes at five resolution corresponding to five telescopes:

vro: pixel_scale = 0.2", fastest run times.
euclid: pixel_scale = 0.1", fast run times
hst: pixel_scale = 0.05", normal run times, represents the type of data we do most our fitting on currently.
hst_up: pixel_scale = 0.03", slow run times.
ao: pixel_scale = 0.01", very slow :(
"""
# instrument = "vro"
# instrument = "euclid"
instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

pixel_scales_dict = {"vro": 0.2, "euclid": 0.1, "hst": 0.05, "hst_up": 0.03, "ao": 0.01}
pixel_scale = pixel_scales_dict[instrument]

"""
Load the dataset for this instrument / resolution.
"""
if on_cosma:
    dataset_path = path.join(
        cosma_path, "dataset", "stochastic", "imaging", "instruments", instrument
    )
else:
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
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    sub_size=sub_size,
    radius=mask_radius,
)

# mask = al.Mask2D.circular_annular(
#     shape_native=imaging.shape_native,
#     pixel_scales=imaging.pixel_scales,
#     sub_size=sub_size,
#     inner_radius=1.0,
#     outer_radius=3.0,
# )

masked_imaging = imaging.apply_mask(mask=mask)

masked_imaging = masked_imaging.apply_settings(
    settings=al.SettingsImaging(sub_size=sub_size)
)


"""
__Coefficient__

Determine the regularization coefficient using the correct lens model.
"""


def func(coefficient):

    source_galaxy.pixelization.regularization.coefficient = coefficient
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
    fit = al.FitImaging(
        dataset=masked_imaging,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(use_border=True),
    )

    fom = fit.figure_of_merit

    print(coefficient, fom)

    return -1.0 * fom


print("\nSetting Regularization Coefficient\n")

coefficient = minimize_scalar(func, method="bounded", bounds=[1e-3, 1e3]).x

print(f"coefficient = {coefficient}")

source_galaxy.pixelization.regularization.coefficient = coefficient

"""
Output an image of the fit, so that we can inspect that it fits the data as expected.
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

source_image = source_galaxy_true.image_2d_from(grid=masked_imaging.grid)

tracer.galaxies[1].hyper_galaxy_image = source_image
tracer.galaxies[1].hyper_model_image = source_image

fit = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer,
    settings_pixelization=al.SettingsPixelization(use_border=True),
)

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_fit_imaging", format="png"
    )
)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_imaging_plotter.subplot_fit_imaging()

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_of_plane_1", format="png"
    )
)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_imaging_plotter.subplot_of_planes(plane_index=1)


"""
__Info__

The `info_dict` contains all the key information of the analysis which describes its run times.
"""
info_dict = {}
info_dict["image_pixels"] = masked_imaging.grid.sub_shape_slim
info_dict["stochastic_seed"] = stochastic_seed
info_dict["sub_size"] = sub_size
info_dict["mask_radius"] = mask_radius
info_dict["mesh_shape_2d"] = mesh_shape_2d
info_dict["source_pixels"] = len(fit.inversion.reconstruction)
info_dict["coefficient"] = coefficient

print(info_dict)

with open(path.join(file_path, f"{instrument}_info.json"), "w+") as outfile:
    json.dump(info_dict, outfile, indent=4)


"""
__Output__

Output the log likelihood values and other terms in the likelihood function input a .json file so they can be
loaded and plotted in other scripts.
"""
if not os.path.exists(file_path):
    os.makedirs(file_path)

filename = f"{instrument}_stochastic_dict.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

"""
__Stochastic Calculation__

Create new Tracer with variable density slope and fit for each lens model, storing all likelihood terms.
"""
slope_total = int(1 + (slope_upper - slope_lower) / slope_interval)
slope_list = list(np.linspace(slope_lower, slope_upper, slope_total))

stochastic_dict = {
    "slope_list": [],
    "figure_of_merit_list": [],
    "chi_squared_list": [],
    "regularization_term_list": [],
    "log_det_curvature_reg_matrix_term_list": [],
    "log_det_regularization_matrix_term_list": [],
    "noise_normalization_list": [],
}

output_interval = 100
counter = 0

for i, slope in enumerate(slope_list):

    counter += 1

    lens_galaxy.mass.slope = slope

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(
        dataset=masked_imaging,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(use_border=True),
    )

    stochastic_dict["slope_list"].append(slope)
    stochastic_dict["figure_of_merit_list"].append(fit.figure_of_merit)
    stochastic_dict["chi_squared_list"].append(fit.chi_squared)
    stochastic_dict["regularization_term_list"].append(
        fit.inversion.regularization_term
    )
    stochastic_dict["log_det_curvature_reg_matrix_term_list"].append(
        fit.inversion.log_det_curvature_reg_matrix_term
    )
    stochastic_dict["log_det_regularization_matrix_term_list"].append(
        fit.inversion.log_det_regularization_matrix_term
    )
    stochastic_dict["noise_normalization_list"].append(fit.noise_normalization)

    print(
        f"{i}/{slope_total}",
        " {:>14.7f} {:>14.2f} {:>14.2f} {:>14.2f} {:>14.2f} {:>14.2f}       {:>14.6f}".format(
            slope,
            fit.chi_squared,
            fit.inversion.regularization_term,
            fit.inversion.log_det_curvature_reg_matrix_term,
            fit.inversion.log_det_regularization_matrix_term,
            fit.noise_normalization,
            fit.figure_of_merit,
        ),
    )

    if counter == output_interval:

        counter = 0

        with open(path.join(file_path, filename), "w") as outfile:
            json.dump(stochastic_dict, outfile, indent=4)
