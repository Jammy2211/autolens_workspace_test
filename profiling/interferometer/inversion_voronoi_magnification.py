"""
__PROFILING: Interferometer Voronoi Magnification Fit__

This profiling script times how long an `Inversion` takes to fit `Interferometer` data.
"""

import autolens as al
import autolens.plot as aplt

from os import path
import numpy as np
import time

"""
When profiling a function, there is randomness in how long it takes to perform its evaluation. We can repeat the
function call multiple times and take the average run time to get a more realible run-time.
"""

repeats = 3

"""These settings control the run-time of the `Inversion` performed on the `Interferometer` data."""

transformer_class = al.TransformerDFT
use_linear_operators = False

"""Load the strong lens dataset `mass_sie__source_sersic` `from .fits files."""

# dataset_path = path.join("dataset", "interferometer", "mass_sie__source_sersic")
dataset_path = path.join("dataset", "interferometer", "instruments", "sma")

interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
)

"""
Set up the lens and source galaxies used to profile the fit. The lens galaxy uses the true model, whereas the source
galaxy includes the `Pixelization` and `Regularization` we profile.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=1.0),
)

"""
Set up the `MaskedInterferometer` dataset we fit. This includes the `real_space_mask` that the source galaxy's 
`Inversion` is evaluated using via mapping to Fourier space using the `Transformer`.
"""

mask = al.Mask2D.circular(
    shape_native=(256, 256), pixel_scales=0.05, sub_size=1, radius=3.0
)

masked_interferometer = al.MaskedInterferometer(
    interferometer=interferometer,
    real_space_mask=mask,
    visibilities_mask=np.full(
        fill_value=False, shape=interferometer.visibilities.shape
    ),
    settings=al.SettingsMaskedInterferometer(transformer_class=transformer_class),
)

"""Print the size of the real-space mask and number of visiblities, which drive the run-time of the fit."""

print(
    f"Number of points in real space = {masked_interferometer.grid.sub_shape_slim} \n"
)
print(f"Number of visibilities = {masked_interferometer.visibilities.shape_slim}\n")

start_overall = time.time()

"""Time the complete fitting procedure."""

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

start = time.time()
for i in range(repeats):
    fit = al.FitInterferometer(
        masked_interferometer=masked_interferometer,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(
            use_linear_operators=use_linear_operators
        ),
    )

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
