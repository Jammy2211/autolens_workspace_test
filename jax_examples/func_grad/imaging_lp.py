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

import numpy as np
import jax.numpy as jnp
import jax
from jax import grad
from os import path

import autofit as af
import autolens as al
from autoconf import conf

conf.instance["general"]["model"]["ignore_prior_limits"] = True

"""
__Dataset__

Load and plot the galaxy dataset `operated` via .fits files, which we will fit with 
the model.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)


"""
__Mask__

The model-fit requires a `Mask2D` defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.
"""
mask_2d = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask_2d)

dataset = dataset.apply_over_sampling(over_sample_size_lp=1)

positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)

# over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
#     grid=dataset.grid,
#     sub_size_list=[8, 4, 1],
#     radial_list=[0.3, 0.6],
#     centre_list=[(0.0, 0.0)],
# )
#
# dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)
#
dataset.convolver

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a model where:

 - The galaxy's bulge is a parametric `Sersic` bulge [7 parameters]. 
 - The galaxy's point source emission is a parametric operated `Gaussian` centred on the bulge [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)

mass = af.Model(al.mp.Isothermal)

shear = af.Model(al.mp.ExternalShear)


lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

# Source:

bulge = af.Model(al.lp_linear.Sersic)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
can compute its gradient.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    #   settings_inversion=al.SettingsInversion(use_positive_only_solver=False)
)


"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**, which pairs
the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
from autofit.non_linear.fitness import Fitness
import time

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

# batch_size = 5
#
# parameters = np.zeros((batch_size, model.total_free_parameters))
#
# for i in range(batch_size):
#     parameters[i, :] = model.random_unit_vector_within_limits()
#
# param_vector = model.physical_values_from_prior_medians
#
# func = fitness
# func.call(param_vector)
# start = time.time()
# for i in range(batch_size):
#     print(func.call(parameters[i, :]))
# print("NO JAX Time taken:", time.time() - start)
#
# func = jax.vmap(fitness)
# print(func(parameters))
#
# start = time.time()
# print(func(jnp.array(parameters2)))
# print("JAX Vmap Time taken:", time.time() - start)

# fitness = Fitness(
#     model=model,
#     analysis=analysis,
#     fom_is_log_likelihood=True,
#     resample_figure_of_merit=-1.0e99,
# )
#
# parameters2 = np.zeros((batch_size, model.total_free_parameters))
#
# for i in range(batch_size):
#     parameters2[i, :] = model.random_unit_vector_within_limits()
#
# parameters2 = jnp.array(parameters2)
#
# param_vector = jnp.array(model.physical_values_from_prior_medians)
#
# fitness._call(param_vector)
# start = time.time()
# for i in range(batch_size):
#     print(fitness._call(jnp.array(parameters2[i, :])))
# print("JAX JIT LOOP Time taken:", time.time() - start)


from jax import profiler


param_vector = jnp.array(model.physical_values_from_prior_medians)
print(fitness.call_numpy_wrapper(param_vector))

start = time.time()

# profiler.start_trace("profiler_output")

print(fitness.call_numpy_wrapper(param_vector))

# profiler.stop_trace()

print("JAX JIT LOOP Time taken:", time.time() - start)

# param_vector = model.physical_values_from_prior_medians
# func = fitness

# start = time.time()
# for i in range(batch_size):
#     func(param_vector)
# print("NO JAX Time taken:", time.time() - start)

# func = jax.jit(fitness)
# start = time.time()
# for i in range(batch_size):
#     func(param_vector)
# print("JAX JIT LOOP Time taken:", time.time() - start)
#
#
#
# def prior_transform_vectorized(cube, model):
#
#     trans = np.array([model.vector_from_unit_vector(row) for row in cube])
#
#     return trans
#
#
# start = time.time()
# prior_transform_vectorized(parameters, model)
# end = time.time()
# print(f"Time taken for prior transform vectorized: {end - start:.4f} seconds")
#
#
# def prior_transform(cube, model):
#     return model.vector_from_unit_vector(unit_vector=cube)
#
#
# start = time.time()
# for cube in parameters:
#     prior_transform(cube, model)
# end = time.time()
#
# print(f"Time taken for prior NORMAL transformed: {end - start:.4f} seconds")
#
#
#
#
# from evosax.algorithms import CMA_ES
#
# solution = jnp.array(model.physical_values_from_prior_medians)
#
# # Instantiate the search strategy
# population_size = batch_size
# es = CMA_ES(population_size=population_size, solution=solution)
# params = es.default_params
#
# # Initialize state
# key = jax.random.key(0)
# key, subkey = jax.random.split(key)
# state = es.init(key, solution, params)
#
# # Ask-Eval-Tell loop
# num_generations = 1000
#
# import time
#
# for i in range(num_generations):
#
#     key, key_ask, key_eval = jax.random.split(key, 3)
#
#     # Generate a set of candidate solutions to evaluate
#     population, state = es.ask(key_ask, state, params)
#
#     start = time.time()
#
#     # Evaluate the fitness of the population
#     fitness = func(population)
#
#     print("fitness time:", time.time() - start)
#
#     # Update the evolution strategy
#     state, metrics = es.tell(key, population, fitness, state, params)
#
# instance = model.instance_from_vector(vector=state.best_solution)
