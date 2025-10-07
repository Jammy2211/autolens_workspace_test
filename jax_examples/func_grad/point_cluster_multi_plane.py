"""
Func Grad: Light Parametric Operated
====================================

This script test if JAX can successfully compute the gradient of the log likelihood of an `Imaging` dataset with a
model which uses operated light profiles.

 __Operated Fitting__

It is common for galaxies to have point-source emission, for example bright emission right at their centre due to
an active galactic nuclei or very compact knot of star formation.

This point-source emission is subject to blurring during data accquisiton due to the telescope optics, and therefore
is not seen as a single pixel of light but spread over multiple pixels as a convolution with the telescope
Point Spread Function (PSF).

It is difficult to model this compact point source emission using a point-source light profile (or an extremely
compact Gaussian / Sersic profile). This is because when the model-image of a compact point source of light is
convolved with the PSF, the solution to this convolution is extremely sensitive to which pixel (and sub-pixel) the
compact model emission lands in.

Operated light profiles offer an alternative approach, whereby the light profile is assumed to have already been
convolved with the PSF. This operated light profile is then fitted directly to the point-source emission, which as
discussed above shows the PSF features.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import copy
import numpy as np
import jax.numpy as jnp
import jax
from jax import grad
from pathlib import Path

import autofit as af
import autolens as al
from autoconf import conf

conf.instance["general"]["model"]["ignore_prior_limits"] = True


"""
__Dataset__

Load the strong lens point-source dataset `simple`, which is the dataset we will use to perform point source 
lens modeling.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "point_source" / dataset_name

"""
We now load the point source dataset we will fit using point source modeling. 

We load this data as a `PointDataset`, which contains the positions of every point source. 
"""
dataset = al.from_json(
    file_path=dataset_path / "point_dataset_positions_only.json",
)

# dataset = al.from_json(
#     file_path=dataset_path / "point_dataset_with_fluxes_and_time_delays.json",
# )


"""
__Point Solver__

For point-source modeling we require a `PointSolver`, which determines the multiple-images of the mass model for a 
point source at location (y,x) in the source plane. 

It does this by ray tracing triangles from the image-plane to the source-plane and calculating if the 
source-plane (y,x) centre is inside the triangle. The method gradually ray-traces smaller and smaller triangles so 
that the multiple images can be determine with sub-pixel precision.

The `PointSolver` requires a starting grid of (y,x) coordinates in the image-plane which defines the first set
of triangles that are ray-traced to the source-plane. It also requires that a `pixel_scale_precision` is input, 
which is the resolution up to which the multiple images are computed. The lower the `pixel_scale_precision`, the
longer the calculation, with the value of 0.001 below balancing efficiency with precision.

Strong lens mass models have a multiple image called the "central image". However, the image is nearly always 
significantly demagnified, meaning that it is not observed and cannot constrain the lens model. As this image is a
valid multiple image, the `PointSolver` will locate it irrespective of whether its so demagnified it is not observed.
To ensure this does not occur, we set a `magnification_threshold=0.1`, which discards this image because its
magnification will be well below this threshold.

If your dataset contains a central image that is observed you should reduce to include it in
the analysis.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 - The source galaxy's light is a point `Point` [2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.

__Name Pairing__

Every point-source dataset in the `PointDataset` has a name, which in this example was `point_0`. This `name` pairs 
the dataset to the `Point` in the model below. Because the name of the dataset is `point_0`, the 
only `Point` object that is used to fit it must have the name `point_0`.

If there is no point-source in the model that has the same name as a `PointDataset`, that data is not used in
the model-fit. If a point-source is included in the model whose name has no corresponding entry in 
the `PointDataset` it will raise an error.

In this example, where there is just one source, name pairing appears unnecessary. However, point-source datasets may
have many source galaxies in them, and name pairing is necessary to ensure every point source in the lens model is 
fitted to its particular lensed images in the `PointDataset`.

__Coordinates__

The model fitting default settings assume that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/*/data_preparation`). 
 - Manually override the lens model priors (`autolens_workspace/*/modeling/imaging/customize/priors.py`).
"""
import pandas as pd

df = pd.read_csv("mass_profiles.csv")

lens_galaxies = {}

i = 0

for _, row in df.iterrows():
    # mass = al.mp.dPIE(
    #     centre=(row["center_y(arcsec)"], row["center_x(arcsec)"]),
    #     ell_comps=(0.0, 0.0),
    #     ra=row["r_core(arcsec)"],
    #     rs=row["r_cut(arcsec)"],
    #     kappa_scale=row["kappa_scale"],
    # )

    mass = af.Model(al.mp.Isothermal)

    mass.centre.centre_0 = row["center_y(arcsec)"]
    mass.centre.centre_1 = row["center_x(arcsec)"]
    mass.ell_comps.ell_comps_0 = 0.01
    mass.ell_comps.ell_comps_1 = 0.01
    mass.einstein_radius = row["kappa_scale"] / (834.6912404)

    lens_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    lens_galaxies[f"lens_{i}"] = lens_galaxy

    i += 1

# centre_list = [
#     (-1.4438660221815516,-1.7620111013273947),
#     (0.5797711347134087,0.8484551909969554),
#     (1.2297902805737424,-0.6864879131353183),
#     (-2.784087133980032,2.598101647247505),
#     (0.1014684665073109,-0.600873706602864),
#     (1.083060436161219,-1.1275136683043994)
# ]

centre_list = [
    (-1.762011101, -1.4438660),
    (0.848455191, 0.5797711),
    (-0.686487913, 1.2297902),
    (2.598101647, -2.7840871),
    (-0.600873707, 0.1014684),
    (-1.127513668, 1.0830604),
]

points = {}

for i, centre in enumerate(centre_list):

    # Source:

    point = af.Model(al.ps.Point)
    point.centre.centre_0 = af.UniformPrior(
        lower_limit=centre[0] - 0.01,
        upper_limit=centre[0] + 0.01,
    )
    point.centre.centre_1 = af.UniformPrior(
        lower_limit=centre[1] - 0.01,
        upper_limit=centre[1] + 0.01,
    )

    points[f"point_{i}"] = point

redshift_list = [1.1778647, 2.100664, 2.7937636, 2.4445848, 4.44669, 1.4089019]

source_galaxies = {}

for i in range(len(centre_list)):

    source_galaxies[f"source_{i}"] = af.Model(
        al.Galaxy, redshift=redshift_list[i], **{f"point_{i}": points[f"point_{i}"]}
    )

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(**lens_galaxies, **source_galaxies))


"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
can compute its gradient.
"""


analysis_factor_list = []

for i in range(len(centre_list)):

    dataset_analysis = copy.copy(dataset)

    dataset_analysis.name = f"point_{i}"

    model_analysis = model.copy()
    analysis = al.AnalysisPoint(dataset=dataset_analysis, solver=solver)
    #
    # instance = model.instance_from_prior_medians()
    # tracer = analysis.tracer_via_instance_from(instance=instance)
    #
    # positions = solver.solve(
    #     tracer=tracer, source_plane_coordinate=centre_list[i]
    # )
    #
    # print(positions)

    analysis_factor = af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

factor_graph = af.FactorGraphModel(*analysis_factor_list)

# print(positions)

"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**, which pairs
the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
import time
from autofit.non_linear.fitness import Fitness

fitness = Fitness(
    model=factor_graph.global_prior_model,
    analysis=factor_graph,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

"""
We now test the JAX-ing of this LH function.
"""
start = time.time()

param_vector = jnp.array(model.physical_values_from_prior_medians)
print(fitness.call_numpy_wrapper(param_vector))

print("JAX Tracing (And LH) Time:", time.time() - start)


print()

start = time.time()

# profiler.start_trace("profiler_output")

print(fitness.call_numpy_wrapper(param_vector))

print("JAX JIT LOOP Time taken:", time.time() - start)


# """
# __Analysis__
#
# The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
# can compute its gradient.
# """
#
#
# analysis_factor_list = []
#
# for i in range(len(centre_list)):
#
#     dataset.name = f"point_{i}"
#
#     model_analysis = model.copy()
#     analysis = al.AnalysisPoint(dataset=dataset, solver=solver)
#
#     instance = model.instance_from_prior_medians()
#     tracer = analysis.tracer_via_instance_from(instance=instance)
#
#     positions = solver.solve(
#         tracer=tracer, source_plane_coordinate=centre_list[i]
#     )
#
#     print(positions)
#
#     analysis_factor = af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)
#
#     analysis_factor_list.append(analysis_factor)
#
# factor_graph = af.FactorGraphModel(*analysis_factor_list)
#
# from autofit.non_linear.fitness import Fitness
# import time
#
# fitness = Fitness(
#     model=model,
#     analysis=analysis,
#     fom_is_log_likelihood=True,
#     resample_figure_of_merit=-1.0e99,
# )
#
# batch_size = 30
#
# parameters = np.zeros((batch_size, model.total_free_parameters))
#
# for i in range(batch_size):
#     parameters[i, :] = model.random_unit_vector_within_limits()
#
# param_vector = model.physical_values_from_prior_medians
#
# result_list = []
#
# # func = fitness
# # func.call(param_vector)
# # start = time.time()
# # for i in range(batch_size):
# #
# #     result = func.call(parameters[i, :])
# #
# #     result_list.append(result)
# #
# # print(result_list)
# # print("NO JAX Time taken:", time.time() - start)
#
# # func = jax.vmap(fitness)
# # print(func(parameters))
# #
# # start = time.time()
# # print(func(jnp.array(parameters)))
# # print("JAX Vmap Time taken:", time.time() - start)
#
# fitness = Fitness(
#     model=model,
#     analysis=analysis,
#     fom_is_log_likelihood=True,
#     resample_figure_of_merit=-1.0e99,
# )
#
# parameters2 = np.array(parameters)
#
# param_vector = np.array(model.physical_values_from_prior_medians)
#
# result_list = []
#
# vectorized_fitness = jax.jit(jax.vmap(fitness.call))
# result = vectorized_fitness(parameters)
#
# # jax.vmap(fitness.call_numpy_wrapper(param_vector)
# start = time.time()
# # for i in range(batch_size):
# #    result = fitness.call_numpy_wrapper(np.array(parameters2[i, :]))
# result = vectorized_fitness(parameters)
#
# result_list.append(result)
#
# print(result)
# print("JAX JIT LOOP Time taken:", time.time() - start)
