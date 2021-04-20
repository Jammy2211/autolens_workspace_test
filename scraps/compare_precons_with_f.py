"""
__PROFILING: Interferometer Voronoi Magnification Fit__

This profiling script times how long an `Inversion` takes to fit `Interferometer` data.
"""
from autoarray.structures import visibilities as vis
from autoarray.operators import transformer as trans
from autoarray.inversion import (
    regularization as reg,
    mappers,
    inversions as inv,
    inversion_util,
)
from os import path
import autolens as al
import numpy as np

"""
When profiling a function, there is randomness in how long it takes to perform its evaluation. We can repeat the
function call multiple times and take the average run time to get a more realible run-time.
"""

repeats = 1

"""Load the strong lens dataset `mass_sie__source_sersic` `from .fits files."""

dataset_name = "sma"
uv_wavelengths_name = "alma_uv_wavelengths_x10k"
dataset_path = path.join("dataset", "instruments", dataset_name)

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
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=45.0),
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

coefficients = list(np.logspace(np.log10(1e-6), np.log10(1e12), 19))

"""
"""

"""These settings control the run-time of the `Inversion` performed on the `Interferometer` data."""

transformer_class = al.TransformerDFT
use_linear_operators = False

interferometer = al.MaskedInterferometer(
    interferometer=interferometer,
    real_space_mask=mask,
    visibilities_mask=np.full(
        fill_value=False, shape=interferometer.visibilities.shape
    ),
    settings=al.SettingsInterferometer(transformer_class=transformer_class),
)

coefficients_matrix_dft = []
terms_matrix_dft = []


def compute_curvature_matrix(
    visibilities: vis.Visibilities,
    noise_map: vis.VisibilitiesNoiseMap,
    transformer: trans.TransformerNUFFT,
    mapper: mappers.Mapper,
    regularization: reg.Regularization,
    settings=inv.SettingsInversion(),
):

    transformed_mapping_matrices = transformer.transformed_mapping_matrices_from_mapping_matrix(
        mapping_matrix=mapper.mapping_matrix
    )

    real_data_vector = inversion_util.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrices[0],
        visibilities=visibilities[:, 0],
        noise_map=noise_map[:, 0],
    )

    imag_data_vector = inversion_util.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrices[1],
        visibilities=visibilities[:, 1],
        noise_map=noise_map[:, 1],
    )

    real_curvature_matrix = inversion_util.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=transformed_mapping_matrices[0], noise_map=noise_map[:, 0]
    )

    imag_curvature_matrix = inversion_util.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=transformed_mapping_matrices[1], noise_map=noise_map[:, 1]
    )

    regularization_matrix = regularization.regularization_matrix_from_mapper(
        mapper=mapper
    )

    data_vector = np.add(real_data_vector, imag_data_vector)
    curvature_matrix = np.add(real_curvature_matrix, imag_curvature_matrix)
    curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

    return curvature_reg_matrix


print("Matrix DFT:")

import matplotlib.pyplot as plt

for coefficient in coefficients:

    reg = al.reg.Constant(coefficient=coefficient)

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
        regularization=reg,
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    mapper = tracer.mappers_of_planes_from_grid(
        grid=interferometer.grid, settings_pixelization=al.SettingsPixelization()
    )[1]

    curvature_matrix = compute_curvature_matrix(
        visibilities=interferometer.visibilities,
        noise_map=interferometer.noise_map,
        transformer=interferometer.transformer,
        mapper=mapper,
        regularization=reg,
    )

    regularization_matrix = reg.regularization_matrix_from_mapper(mapper=mapper)

    preconditioner_matrix = inversion_util.preconditioner_matrix_via_mapping_matrix_from(
        mapping_matrix=mapper.mapping_matrix,
        regularization_matrix=regularization_matrix,
        preconditioner_noise_normalization=np.sum(1.0 / interferometer.noise_map ** 2),
    )

    preconditioner_inv = np.linalg.inv(preconditioner_matrix)

    print(curvature_matrix)
    print(preconditioner_inv)

    plt.figure(figsize=(16, 4.5))
    plt.set_cmap("jet")
    plt.subplot(1, 3, 1)
    plt.imshow(curvature_matrix)
    plt.title(f"Curvature Reg matrix (F + lambda H), Coeff = {coefficient}")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.colorbar()
    plt.imshow(preconditioner_inv)
    plt.title(f"P^-1,, Coeff = {coefficient}")
    plt.subplot(1, 3, 3)
    plt.colorbar()
    plt.title(f"Curvature Reg matrix / P^-1, Coeff = {coefficient}")
    plt.imshow(curvature_matrix / preconditioner_inv)
    plt.show()
    plt.close()
