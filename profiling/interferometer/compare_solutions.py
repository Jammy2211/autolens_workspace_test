"""
__PROFILING: Interferometer Voronoi Magnification Fit__

This profiling script times how long an `Inversion` takes to fit `Interferometer` data.
"""
from os import path
import autolens as al
import numpy as np

"""
When profiling a function, there is randomness in how long it takes to perform its evaluation. We can repeat the
function call multiple times and take the average run time to get a more realible run-time.
"""

repeats = 1

"""Load the strong lens dataset `mass_sie__source_sersic` `from .fits files."""

dataset_name = "alma"
uv_wavelengths_name = "alma_uv_wavelengths_x10k"
dataset_path = path.join("dataset", "instruments", dataset_name, uv_wavelengths_name)

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

"""
Set up the `MaskedInterferometer` dataset we fit. This includes the `real_space_mask` that the source galaxy's 
`Inversion` is evaluated using via mapping to Fourier space using the `Transformer`.
"""

mask = al.Mask2D.circular(
    shape_native=(251, 251), pixel_scales=0.05, sub_size=1, radius=4.5
)

"""The coefficients the plot is made using."""

"""
"""

"""These settings control the run-time of the `Inversion` performed on the `Interferometer` data."""

transformer_class = al.TransformerDFT
use_linear_operators = False
use_preconditioner = False

masked_interferometer = al.MaskedInterferometer(
    interferometer=interferometer,
    real_space_mask=mask,
    visibilities_mask=np.full(
        fill_value=False, shape=interferometer.visibilities.shape
    ),
    settings=al.SettingsMaskedInterferometer(transformer_class=transformer_class),
)

coefficients_matrix_dft = []
log_det_curvature_reg_matrix_terms_matrix_dft = []
evidence_matrix_dft = []

print("Matrix DFT:")

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=1.0),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit_dft = al.FitInterferometer(
    masked_interferometer=masked_interferometer,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(
        use_linear_operators=use_linear_operators, use_preconditioner=use_preconditioner
    ),
)


"""
"""

"""Repeat for NUFFT via Matrices"""

transformer_class = al.TransformerNUFFT
use_linear_operators = True
use_preconditioner = False

masked_interferometer = al.MaskedInterferometer(
    interferometer=interferometer,
    real_space_mask=mask,
    visibilities_mask=np.full(
        fill_value=False, shape=interferometer.visibilities.shape
    ),
    settings=al.SettingsMaskedInterferometer(transformer_class=transformer_class),
)

print("Matrix NUFFT:")

coefficients_matrix_nufft = []
chi_squared_terms_matrix_nufft = []
regularization_terms_matrix_nufft = []
log_det_curvature_reg_matrix_terms_matrix_nufft = []
log_det_regularization_terms_matrix_nufft = []
evidence_matrix_nufft = []
max_source_flux_matrix_nufft = []

coefficients = list(np.logspace(np.log10(0.001), np.log10(10.0), 50))

for coefficient in coefficients:

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
        regularization=al.reg.Constant(coefficient=coefficient),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitInterferometer(
        masked_interferometer=masked_interferometer,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(
            use_linear_operators=use_linear_operators,
            use_preconditioner=use_preconditioner,
        ),
    )

    chi_squared_terms_matrix_nufft.append(fit_dft.chi_squared - fit.chi_squared)
    regularization_terms_matrix_nufft.append(
        fit_dft.inversion.regularization_term - fit.inversion.regularization_term
    )
    log_det_curvature_reg_matrix_terms_matrix_nufft.append(
        float(fit_dft.inversion.log_det_curvature_reg_matrix_term)
        - float(fit.inversion.log_det_curvature_reg_matrix_term)
    )
    log_det_regularization_terms_matrix_nufft.append(
        fit_dft.inversion.log_det_regularization_matrix_term
        - fit.inversion.log_det_regularization_matrix_term
    )
    evidence_matrix_nufft.append(fit_dft.figure_of_merit - fit.figure_of_merit)
    max_source_flux_matrix_nufft.append(
        np.max(fit_dft.inversion.reconstruction) - np.max(fit.inversion.reconstruction)
    )

    coefficients_matrix_nufft.append(coefficient)

    print(
        coefficient,
        chi_squared_terms_matrix_nufft[-1],
        regularization_terms_matrix_nufft[-1],
        log_det_curvature_reg_matrix_terms_matrix_nufft[-1],
        evidence_matrix_nufft[-1],
        max_source_flux_matrix_nufft[-1],
    )
