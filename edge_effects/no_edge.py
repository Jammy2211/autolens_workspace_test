"""
__PROFILING: Inversion DelaunayMagnification__

This profiling script times how long it takes to fit `Imaging` data with a `DelaunayMagnification` mesh for
datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoLens** log likelihood function.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import numpy as np
import time
import json

import autolens as al
import autolens.plot as aplt


"""
The path all profiling results are output.
"""
profiling_path = path.dirname(path.realpath(__file__))

file_path = os.path.join(profiling_path, "output", "edge")

"""
Whether w_tilde is used dictates the output folder.
"""
use_w_tilde = False
use_positive_only_solver = True
force_edge_pixels_to_zeros = True
# force_edge_pixels_to_zeros = False

settings_inversion = al.SettingsInversion(
    use_w_tilde=use_w_tilde,
    use_positive_only_solver=use_positive_only_solver,
    force_edge_pixels_to_zeros=force_edge_pixels_to_zeros
)


"""
The number of repeats used to estimate the run time.
"""
repeats = 1

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 4
mask_radius = 3.5
psf_shape_2d = (21, 21)
mesh_shape_2d = (40, 40)

print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"psf shape = {psf_shape_2d}")
print(f"pixelization shape = {mesh_shape_2d}")

total_gaussians = 30

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

gaussian_list = []

for i, gaussian_index in enumerate(range(total_gaussians)):

    gaussian = al.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.2),
        sigma=10 ** log10_sigma_list[i],
    )

    gaussian_list.append(gaussian)

lens_bulge = al.lp_basis.Basis(
    light_profile_list=gaussian_list,
    regularization=al.reg.ConstantZeroth(
        coefficient_neighbor=0.0, coefficient_zeroth=1.0e-3
    ),
)

gaussian_list = []

for i, gaussian_index in enumerate(range(total_gaussians)):

    gaussian = al.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(-0.3, 0.4),
        sigma=10 ** log10_sigma_list[i],
    )

    gaussian_list.append(gaussian)

lens_disk = al.lp_basis.Basis(
    light_profile_list=gaussian_list,
    regularization=al.reg.ConstantZeroth(
        coefficient_neighbor=0.0, coefficient_zeroth=1.0e-3
    ),
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=lens_bulge,
    disk=lens_disk,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.3,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

"""
The source galaxy whose `DelaunayMagnification` `Pixelization` fits the data.
"""
mesh = al.mesh.VoronoiNNMagnification(shape=mesh_shape_2d)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.Pixelization(
        mesh=mesh, regularization=al.reg.ConstantSplit(coefficient=5.0)
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
pixel_scale = 0.05

"""
Load the dataset for this instrument / resolution.
"""
dataset_path = path.join("dataset", "imaging", "edge_effects")

imaging = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
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

masked_imaging = imaging.apply_mask(mask=mask)

masked_imaging = masked_imaging.apply_settings(
    settings=al.SettingsImaging(sub_size_pixelization=sub_size)
)

"""
__Numba Caching__

Call FitImaging once to get all numba functions initialized.
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer,
    settings_inversion=settings_inversion,
)
print(fit.figure_of_merit)

"""
__Fit Time__

Time FitImaging by itself, to compare to profiling dict call.
"""
start = time.time()
for i in range(repeats):
    fit = al.FitImaging(
        dataset=masked_imaging,
        tracer=tracer,
        settings_inversion=settings_inversion,
    )
    fit.log_evidence
fit_time = (time.time() - start) / repeats
print(f"Fit Time = {fit_time} \n")


"""
__Profiling Dict__

Apply mask, settings and profiling dict to fit, such that timings of every individiual function are provided.
"""
profiling_dict = {}

tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy], profiling_dict=profiling_dict
)

fit = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer,
    settings_inversion=settings_inversion,
    profiling_dict=profiling_dict,
)
fit.figure_of_merit

profiling_dict = fit.profiling_dict

"""
__Results__

These two numbers are the primary driver of run time. More pixels = longer run time.
"""
print(f"Number of pixels = {masked_imaging.grid.shape_slim} \n")
print(f"Number of sub-pixels = {masked_imaging.grid.sub_shape_slim} \n")

"""
Print the profiling results of every step of the fit for command line output when running profiling scripts.
"""
for key, value in profiling_dict.items():
    print(key, value)

"""
__Predicted And Exccess Time__

The predicted time is how long we expect the fit should take, based on the individual profiling of functions above.

The excess time is the difference of this value from the fit time, and it indiciates whether the break-down above
has missed expensive steps.
"""
predicted_time = 0.0
predicted_time = sum(profiling_dict.values())
excess_time = fit_time - predicted_time

print(f"\nExcess Time = {excess_time} \n")

"""
__Output__

Output the profiling run times as a dictionary so they can be used in `profiling/graphs.py` to create graphs of the
profile run times.

This is stored in a folder using the **PyAutoLens** version number so that profiling run times can be tracked through
**PyAutoLens** development.
"""
if not os.path.exists(file_path):
    os.makedirs(file_path)

filename = f"profiling_dict.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(profiling_dict, outfile)

"""
Output the profiling run time of the entire fit.
"""
filename = f"fit_time.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(fit_time, outfile)

"""
Output an image of the fit, so that we can inspect that it fits the data as expected.
"""
mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"subplot_fit", format="png"
    )
)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_imaging_plotter.subplot_fit()

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"subplot_of_plane_1", format="png"
    )
)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_imaging_plotter.subplot_of_planes(plane_index=1)

"""
The `info_dict` contains all the key information of the analysis which describes its run times.
"""
info_dict = {}
info_dict["repeats"] = repeats
info_dict["image_pixels"] = masked_imaging.grid.sub_shape_slim
info_dict["sub_size"] = sub_size
info_dict["mask_radius"] = mask_radius
info_dict["psf_shape_2d"] = psf_shape_2d
try:
    info_dict[
        "w_tilde_curvature_preload_size"
    ] = fit.inversion.leq.w_tilde.curvature_preload.shape[0]
except AttributeError:
    pass
info_dict["source_pixels"] = len(fit.inversion.reconstruction)
info_dict["excess_time"] = excess_time

print(info_dict)

with open(path.join(file_path, f"info.json"), "w+") as outfile:
    json.dump(info_dict, outfile, indent=4)

print(fit.inversion.reconstruction[fit.inversion.mapper_edge_pixel_list])

print(f"Chi Squared: {fit.chi_squared}")
print(f"Regularization Term: {fit.inversion.regularization_term}")
print(f"Log Det Curvature Reg Matrix Term: {fit.inversion.log_det_curvature_reg_matrix_term}")
print(f"Log Det Regularization Matrix Term: {fit.inversion.log_det_regularization_matrix_term}")
print(f"Noise Normalization: {fit.noise_normalization}")
print(f"Figure of Merit: {fit.figure_of_merit}")