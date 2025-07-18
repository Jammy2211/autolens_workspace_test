"""
__PROFILING: Inversion Delaunay__

This profiling script times how long it takes to fit `Imaging` data with a `Delaunay` mesh for
datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoLens** log likelihood function.
"""

import os
from os import path
import numpy as np

import time
import json

import autolens as al
import autolens.plot as aplt


"""
The path all profiling results are output.
"""
profiling_path = path.dirname(path.realpath(__file__))

file_path = os.path.join(profiling_path, "times", al.__version__)

"""
The number of repeats used to estimate the run time.
"""
repeats = conf.instance["general"]["profiling"]["repeats"]
print("Number of repeats = " + str(repeats))
print()

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 4
mask_radius = 3.5
psf_shape_2d = (21, 21)
mesh_shape_2d = (35, 35)

print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"psf shape = {psf_shape_2d}")
print(f"pixelization shape = {mesh_shape_2d}")


total_gaussians = 180

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".

mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# A list of linear light profile Gaussians will be input here, which will then be used to fit the data.

basis_gaussian_list = []

# Iterate over every Gaussian and create it, with it centered at (0.0", 0.0") and assuming spherical symmetry.

for i in range(total_gaussians):
    gaussian = al.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.052, 0.0),
        sigma=10 ** log10_sigma_list[i],
    )

    basis_gaussian_list.append(gaussian)

basis = al.lp_basis.Basis(profile_list=basis_gaussian_list)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=basis,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.001, gamma_2=0.001),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.Pixelization(
        image_mesh=al.image_mesh.Overlay(shape=mesh_shape_2d),
        mesh=al.mesh.Delaunay(),
        regularization=al.reg.ConstantSplit(coefficient=1.0),
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
dataset_path = path.join("dataset", "imaging", "instruments", instrument)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scale,
    # over_sampling=al.OverSamplingDataset(
    #     pixelization=4
    # ),
)

dataset.psf = dataset.psf.resized_from(new_shape=psf_shape_2d)

"""
Apply the 2D mask, which for the settings above is representative of the masks we typically use to model strong lenses.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

# mask = al.Mask2D.circular_annular(
#     shape_native=dataset.shape_native,
#     pixel_scales=dataset.pixel_scales,
#     sub_size=sub_size,
#     inner_radius=1.5,
#     outer_radius=2.5,
# )

masked_dataset = dataset.apply_mask(mask=mask)

"""
__Numba Caching__

Call FitImaging once to get all numba functions initialized.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(
    dataset=masked_dataset,
    tracer=tracer,
)
print(fit.figure_of_merit)


"""
__Fit Time__

Time FitImaging by itself, to compare to profiling dict call.
"""
start = time.time()
for i in range(repeats):
    fit = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer,
    )
    fit.log_evidence
fit_time = (time.time() - start) / repeats
print(f"Fit Time = {fit_time} \n")


"""
__Profiling Dict__

Apply mask, settings and profiling dict to fit, such that timings of every individiual function are provided.
"""
run_time_dict = {}

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy], run_time_dict=run_time_dict)

fit = al.FitImaging(
    dataset=masked_dataset,
    tracer=tracer,
    run_time_dict=run_time_dict,
)

fit.figure_of_merit

run_time_dict = fit.run_time_dict

"""
__Results__

These two numbers are the primary driver of run time. More pixels = longer run time.
"""

print(f"Inversion fit run times for image type {instrument} \n")
print(f"Number of pixels = {masked_dataset.grid.shape_slim} \n")
# print(f"Number of sub-pixels = {masked_dataset.grid.over_sampler.sub_total} \n")

"""
Print the profiling results of every step of the fit for command line output when running profiling scripts.
"""
for key, value in run_time_dict.items():
    print(key, value)

"""
__Predicted And Exccess Time__

The predicted time is how long we expect the fit should take, based on the individual profiling of functions above.

The excess time is the difference of this value from the fit time, and it indicates whether the break-down above
has missed expensive steps.
"""
predicted_time = 0.0
predicted_time = sum(run_time_dict.values())
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

filename = f"{instrument}_run_time_dict.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(run_time_dict, outfile)

"""
Output the profiling run time of the entire fit.
"""
filename = f"{instrument}_fit_time.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(fit_time, outfile)

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

"""
The `info_dict` contains all the key information of the analysis which describes its run times.
"""
info_dict = {}
info_dict["repeats"] = repeats
info_dict["image_pixels"] = masked_dataset.grid.shape_slim
info_dict["sub_size"] = sub_size
info_dict["mask_radius"] = mask_radius
info_dict["psf_shape_2d"] = psf_shape_2d
try:
    info_dict["w_tilde_curvature_preload_size"] = (
        fit.inversion.leq.w_tilde.curvature_preload.shape[0]
    )
except AttributeError:
    pass
info_dict["source_pixels"] = len(fit.inversion.reconstruction)
info_dict["excess_time"] = excess_time

print(info_dict)

with open(path.join(file_path, f"{instrument}_info.json"), "w+") as outfile:
    json.dump(info_dict, outfile, indent=4)


print(f"Chi Squared: {fit.chi_squared}")
print(f"Noise Normalization: {fit.noise_normalization}")
print(f"Figure of Merit: {fit.figure_of_merit}")


print(fit.inversion.operated_mapping_matrix.shape)
print(fit.inversion.data_vector.shape)
print(fit.inversion.curvature_reg_matrix.shape)
print(fit.inversion.regularization_matrix.shape)


np.save("blurred_mapping_matrix", fit.inversion.operated_mapping_matrix)
np.save("data_vector", fit.inversion.data_vector)
np.save("curvature_reg_matrix", fit.inversion.curvature_reg_matrix)
np.save("regularization_matrix", fit.inversion.regularization_matrix)
np.save("reconstruction", fit.inversion.reconstruction)
