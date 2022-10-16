"""
__PROFILING: Interferometer Voronoi Magnification Fit__

This profiling script times how long an `Inversion` takes to fit `Interferometer` data.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", ""))

import autolens as al
import autolens.plot as aplt
from autoarray.inversion import mappers
import json
import time
import numpy as np

"""
The path all profiling results are output.
"""
file_path = os.path.join(
    "interferometer",
    "profiling",
    "times",
    al.__version__,
    "inversion_voronoi_magnification",
)

"""
Whether w_tilde is used dictates the output folder.
"""
use_w_tilde = False
use_linear_operators = False
if use_w_tilde and not use_linear_operators:
    file_path = os.path.join(file_path, "w_tilde")
elif not use_w_tilde and not use_linear_operators:
    file_path = os.path.join(file_path, "mapping")
elif not use_w_tilde and use_linear_operators:
    file_path = os.path.join(file_path, "lop")

"""
The number of repeats used to estimate the `Inversion` run time.
"""
repeats = conf.instance["general"]["profiling"]["repeats"]
print("Number of repeats = " + str(repeats))
print()

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 1
real_shape_native = (800, 800)
pixel_scales = (0.05, 0.05)
mask_radius = 3.5
mesh_shape_2d = (57, 57)

print(f"sub grid size = {sub_size}")
print(f"real space mask shape native = {real_shape_native}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"pixelization shape = {mesh_shape_2d}")

"""
Set up the lens and source galaxies used to profile the fit. The lens galaxy uses the true model, whereas the source
galaxy includes the `Pixelization`  we profile.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

"""
The source galaxy whose `VoronoiMagnification` `Pixelization` fits the data.
"""
mesh = al.mesh.VoronoiMagnification(shape=mesh_shape_2d)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.Pixelization(
        mesh=mesh, regularization=al.reg.Constant(coefficient=1.0)
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

"""
Set up the `Interferometer` dataset we fit. This includes the `real_space_mask` that the source galaxy's 
`Inversion` is evaluated using via mapping to Fourier space using the `Transformer`.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=real_shape_native,
    pixel_scales=pixel_scales,
    sub_size=sub_size,
    radius=mask_radius,
)

"""
Load the strong lens dataset `mass_sie__source_sersic` `from .fits files.
"""
instrument = "sma"

dataset_path = path.join("dataset", "interferometer", "instruments", instrument)

interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

"""
These settings control the run-time of the `Inversion` performed on the `Interferometer` data.
"""
transformer_class = al.TransformerDFT

interferometer = interferometer.apply_settings(
    settings=al.SettingsInterferometer(transformer_class=transformer_class)
)

"""
__Numba Caching__

Call FitImaging once to get all numba functions initialized.
"""
fit = al.FitInterferometer(
    dataset=interferometer,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(
        use_w_tilde=use_w_tilde, use_linear_operators=use_linear_operators
    ),
)
print(fit.figure_of_merit)

"""
__Fit Time__

Time FitImaging by itself, to compare to profiling dict call.
"""
start = time.time()
for i in range(repeats):
    fit = al.FitInterferometer(
        dataset=interferometer,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(
            use_w_tilde=use_w_tilde, use_linear_operators=use_linear_operators
        ),
    )
    fit.figure_of_merit
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

fit = al.FitInterferometer(
    dataset=interferometer,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(
        use_w_tilde=use_w_tilde, use_linear_operators=use_linear_operators
    ),
    profiling_dict=profiling_dict,
)
fit.figure_of_merit

profiling_dict = fit.profiling_dict

"""
__Results__

These two numbers are the primary driver of run time. More pixels = longer run time.
"""

print(f"Inversion fit run times for image type {instrument} \n")
print(f"Number of pixels = {interferometer.grid.shape_slim} \n")
print(f"Number of sub-pixels = {interferometer.grid.sub_shape_slim} \n")

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

filename = f"{instrument}_profiling_dict.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(profiling_dict, outfile)

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
        path=file_path,
        filename=f"{instrument}_subplot_fit_interferometer",
        format="png",
    )
)
fit_interferometer_plotter = aplt.FitInterferometerPlotter(
    fit=fit, mat_plot_2d=mat_plot_2d
)
fit_interferometer_plotter.subplot_fit_interferometer()
fit_interferometer_plotter.subplot_fit_dirty_images()
fit_interferometer_plotter.subplot_fit_real_space()

"""
The `info_dict` contains all the key information of the analysis which describes its run times.
"""
info_dict = {}
info_dict["repeats"] = repeats
info_dict["image_pixels"] = interferometer.grid.sub_shape_slim
info_dict["sub_size"] = sub_size
info_dict["mask_radius"] = mask_radius
# info_dict["source_pixels"] = len(reconstruction)

print(info_dict)

with open(path.join(file_path, f"{instrument}_info.json"), "w") as outfile:
    json.dump(info_dict, outfile)
