# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")
# import jax
#
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# import numpy as np
#
# import jax.numpy as jnp
# from jax import grad

import os

import jax

from os import path

import autofit as af
import autolens as al
from autoconf import conf


def fit():
    """
    __Dataset__

    Load and plot the galaxy dataset `operated` via .fits files, which we will fit with
    the model.
    """
    dataset_name = "simple__no_lens_light"
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

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        #    bulge=bulge,
        mass=mass,
        shear=shear,
    )

    # Source:

    bulge = af.Model(al.lp_linear.Sersic)

    source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

    # Overall Lens Model:

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    """
    __Analysis__
    
    The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
    can compute its gradient.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    )

    search = af.Nautilus(
        name="imaging_lp_jax_dask_multi_futures_batch_300",
        unique_tag=dataset_name,
        n_live=150,
        vectorized=False,
        iterations_per_update=100000,
        #   force_x1_cpu=True,
        number_of_cores=4,
        batch_size=4 * 75,
    )

    result = search.fit(model=model, analysis=analysis)

    # """
    # The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**, which pairs
    # the model with likelihood.
    #
    # This is the function on which JAX gradients are computed, so we create this class here.
    # """
    # from autofit.non_linear.fitness import Fitness
    #
    # fitness = Fitness(
    #     model=model,
    #     analysis=analysis,
    #     fom_is_log_likelihood=True,
    #     convert_to_chi_squared=True,
    #     resample_figure_of_merit=-1.0e99,
    # )

    # """
    # We now test the JAX-ing of this LH function.
    # """
    # func = jax.vmap(fitness)

    # from evosax.algorithms import DiffusionEvolution
    # from evosax.algorithms.population_based.diffusion_evolution import State
    #
    # population_size = 200
    # num_generations = 100
    #
    # population = jnp.zeros((population_size, model.total_free_parameters))
    # population_physical = jnp.zeros((population_size, model.total_free_parameters))
    #
    # for i in range(population_size):
    #
    #     params_vector = jnp.array(model.random_unit_vector_within_limits())
    #     population = population.at[i].set(params_vector)
    #
    #     physical = model.vector_from_unit_vector(params_vector)
    #     population_physical = population_physical.at[i].set(physical)
    #
    # print(population_physical[-1])
    #
    # fitness = func(population_physical)
    #
    # es = DiffusionEvolution(
    #     population_size=population_size,
    #     solution=population,
    #     num_generations=num_generations
    # )
    #
    # params = es.default_params
    #
    # # Initialize state
    # key = jax.random.key(0)
    # key, subkey = jax.random.split(key)
    #
    # best_fitness = jnp.argmax(fitness)
    # best_solution = population[best_fitness]
    #
    # state = State(
    #         population=population,
    #         fitness=fitness,
    #         std=1.0,
    #         latent_projection=jnp.eye(model.total_free_parameters),
    #         best_solution=best_solution,
    #         best_fitness=fitness[best_fitness],
    #         generation_counter=0,
    #     )
    #
    # for i in range(num_generations):
    #
    #     key, key_ask, key_eval = jax.random.split(key, 3)
    #
    #     # Generate a set of candidate solutions to evaluate
    #     population, state = es._ask(key=key_ask, state=state, params=params)
    #
    #     print(population[-1])
    #
    #     # Evaluate the fitness of the population
    #
    #     for i in range(population_size):
    #
    #         physical = model.vector_from_unit_vector(population[i])
    #         population_physical = population_physical.at[i].set(physical)
    #
    #     print(population_physical[-1])
    #     kkk
    #
    #     fitness = func(population_physical)
    #
    #     # Update the evolution strategy
    #     state = es._tell(
    #         key=key,
    #         population=population,
    #         fitness=fitness,
    #         state=state,
    #         params=params
    #     )
    #
    #     print(state.fitness)
    #     print()
    #
    #     instance = model.instance_from_vector(vector=state.best_solution)
    #
    #     print(i*population_size, -0.5 * state.best_fitness, instance.galaxies.lens.mass.einstein_radius)
    #
    # instance = model.instance_from_vector(vector=state.best_solution)
    #
    # print(instance)
    #
    # print(-0.5 * state.best_fitness)

    # from evosax.algorithms import CMA_ES
    #
    # solution = jnp.array(model.physical_values_from_prior_medians)
    #
    # # Instantiate the search strategy
    # population_size = 200
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
    # for i in range(num_generations):
    #
    #     key, key_ask, key_eval = jax.random.split(key, 3)
    #
    #     # Generate a set of candidate solutions to evaluate
    #     population, state = es.ask(key_ask, state, params)
    #
    #     # Evaluate the fitness of the population
    #     fitness = func(population)
    #
    #     # Update the evolution strategy
    #     state, metrics = es.tell(key, population, fitness, state, params)
    #
    #     instance = model.instance_from_vector(vector=state.best_solution)
    #
    #     print(i*population_size, -0.5 * state.best_fitness, instance.galaxies.lens.mass.einstein_radius)
    #
    # instance = model.instance_from_vector(vector=state.best_solution)
    #
    # print(instance)
    #
    # print(-0.5 * state.best_fitness)


if __name__ == "__main__":
    fit()
