"""
__PROFILING: Interferometer Voronoi Magnification Fit__

This profiling script times how long an `Inversion` takes to fit `Interferometer` data.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import autolens as al
import autolens.plot as aplt
import time

"""
When profiling a function, there is randomness in how long it takes to perform its evaluation. We can repeat the
function call multiple times and take the average run time to get a more realible run-time.
"""
repeats = 3

"""
These settings control the run-time of the `Inversion` performed on the `Interferometer` data.
"""
transformer_class = al.TransformerDFT
use_linear_operators = False

"""
Set up the `Interferometer` dataset we fit. This includes the `real_space_mask` that the source galaxy's 
`Inversion` is evaluated using via mapping to Fourier space using the `Transformer`.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256), pixel_scales=0.05, sub_size=1, radius=3.0
)

"""
Load the strong lens dataset `mass_sie__source_sersic` `from .fits files.
"""
# dataset_path = path.join("dataset", "interferometer", "mass_sie__source_sersic")
dataset_path = path.join("dataset", "interferometer", "instruments", "sma")

interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

"""
Set up the lens and source galaxies used to profile the fit. The lens galaxy uses the true model, whereas the source
galaxy includes the `Pixelization` and `Regularization` we profile.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.01),
        intensity=0.1,
        effective_radius=0.2,
        sersic_index=3.0,
    ),
)

interferometer = interferometer.apply_settings(
    settings=al.SettingsInterferometer(transformer_class=transformer_class)
)

"""
Print the size of the real-space mask and number of visiblities, which drive the run-time of the fit.
"""
print(f"Number of points in real space = {interferometer.grid.sub_shape_slim} \n")
print(f"Number of visibilities = {interferometer.visibilities.shape_slim}\n")

start_overall = time.time()

"""
Time the complete fitting procedure.
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

start = time.time()
for i in range(repeats):
    fit = al.FitInterferometer(interferometer=interferometer, tracer=tracer)

calculation_time = time.time() - start
print("Time to compute fit = {}".format(calculation_time / repeats))

print(fit.figure_of_merit)

fit_interferometer_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_interferometer_plotter.subplot_fit_interferometer()
fit_interferometer_plotter.subplot_fit_real_space()

"""Time how long it takes to map the reconstruction of the `Inversion` back to the image-plane visibilities."""

start = time.time()
for i in range(repeats):
    fit.inversion.mapped_reconstructed_visibilities

calculation_time = time.time() - start
print("Time to compute inversion mapped = {}".format(calculation_time / repeats))
