"""
__PROFILING: Interferometer Voronoi Magnification Fit__

This profiling script times how long an `Inversion` takes to fit `Interferometer` data.
"""

import autolens as al
import numpy as np
import time

"""
When profiling a function, there is randomness in how long it takes to perform its evaluation. We can repeat the
function call multiple times and take the average run time to get a more realible run-time.
"""

repeats = 1

"""These settings control the run-time of the `Inversion` performed on the `Interferometer` data."""

transformer_class = al.TransformerNUFFT
use_linear_operators = True
use_preconditioner = True

"""Create the dataset which is an input number of visilbitieis which are just noise."""

total_visibilities = 100000

np.random.seed(seed=1)
visibilities = np.ones(shape=(total_visibilities, 2))
visibilities = np.random.uniform(low=-5.0, high=5.0, size=(total_visibilities, 2))
visibilities = al.Visibilities.manual_slim(visibilities=visibilities)

uv_wavelengths = np.ones(shape=(total_visibilities, 2))
noise_map = al.VisibilitiesNoiseMap.ones(shape_slim=(total_visibilities,))

interferometer = al.Interferometer(
    visibilities=visibilities, noise_map=noise_map, uv_wavelengths=uv_wavelengths
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
    shear=al.mp.ExternalShear(elliptical_comps=(0.0, 0.05)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=100.0),
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

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

"""Time the complete fitting procedure."""

start_overall = time.time()

start = time.time()
for i in range(repeats):
    fit = al.FitInterferometer(
        masked_interferometer=masked_interferometer,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(
            use_linear_operators=use_linear_operators,
            use_preconditioner=use_preconditioner,
        ),
    )

calculation_time = time.time() - start
print("Time to compute fit = {}".format(calculation_time / repeats))

"""Time how long it takes to map the reconstruction of the `Inversion` back to the image-plane visibilities."""

start = time.time()
for i in range(repeats):
    fit.inversion.mapped_reconstructed_visibilities

calculation_time = time.time() - start
print("Time to compute inversion mapped = {}".format(calculation_time / repeats))


fit_no_precon = al.FitInterferometer(
    masked_interferometer=masked_interferometer,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(
        use_linear_operators=use_linear_operators, use_preconditioner=False
    ),
)

print(fit_no_precon.figure_of_merit)

fit_precon = al.FitInterferometer(
    masked_interferometer=masked_interferometer,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(
        use_linear_operators=use_linear_operators, use_preconditioner=True
    ),
)

print(fit_precon.figure_of_merit)
