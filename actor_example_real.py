# file: dask_example_single_vector_actor_sync_result.py

from autoconf import cached_property
import numpy as np
from os import path
import jax
import jax.numpy as jnp

from dask.distributed import Client

import autofit as af
import autolens as al


class Fitness:
    def __init__(self, n_dim: int):
        self.n_dim = n_dim

    def _sum(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x)

    @cached_property
    def single_jit(self):
        print(">>> JAX compile happening now!")
        return jax.jit(self._sum)

    def call(self, params: np.ndarray) -> float:
        arr = jnp.array(params, dtype=jnp.float32)
        return float(self.single_jit(arr))


def main():
    dataset_name = "simple"
    dataset_path = path.join("dataset", "imaging", dataset_name)

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=0.1,
    )

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
    n_dim = 19

    # dataset_name = "source_complex"
    # dataset_path = path.join("dataset", "imaging", dataset_name)
    #
    # dataset = al.Imaging.from_fits(
    #     data_path=path.join(dataset_path, "data.fits"),
    #     psf_path=path.join(dataset_path, "psf.fits"),
    #     noise_map_path=path.join(dataset_path, "noise_map.fits"),
    #     pixel_scales=0.05,
    # )
    #

    # mask_2d = al.Mask2D.circular(
    #     shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.5
    # )
    #
    # dataset = dataset.apply_mask(mask=mask_2d)
    #
    # dataset = dataset.apply_over_sampling(over_sample_size_lp=4)
    #
    # dataset.convolver
    #
    # The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
    # """
    # # Lens:
    #
    # total_gaussians = 30
    # gaussian_per_basis = 2
    #
    # # The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
    # mask_radius = 3.0
    # log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)
    #
    # # By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.
    #
    # centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    # centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    #
    # bulge_gaussian_list = []
    #
    # for j in range(gaussian_per_basis):
    #     # A list of Gaussian model components whose parameters are customized belows.
    #
    #     gaussian_list = af.Collection(
    #         af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    #     )
    #
    #     # Iterate over every Gaussian and customize its parameters.
    #
    #     for i, gaussian in enumerate(gaussian_list):
    #         gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
    #         gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
    #         gaussian.ell_comps = gaussian_list[
    #             0
    #         ].ell_comps  # All Gaussians have same elliptical components.
    #         gaussian.sigma = (
    #                 10 ** log10_sigma_list[i]
    #         )  # All Gaussian sigmas are fixed to values above.
    #
    #     bulge_gaussian_list += gaussian_list
    #
    # # The Basis object groups many light profiles together into a single model component.
    #
    # bulge = af.Model(
    #     al.lp_basis.Basis,
    #     profile_list=bulge_gaussian_list,
    # )
    #
    # mass = af.Model(al.mp.Isothermal)
    #
    # shear = af.Model(al.mp.ExternalShear)
    #
    # lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)
    #
    # # Source:
    #
    # total_gaussians = 20
    # gaussian_per_basis = 1
    #
    # # By defining the centre here, it creates two free parameters that are assigned to the source Gaussians.
    #
    # centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    # centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    #
    # log10_sigma_list = np.linspace(-2, np.log10(1.0), total_gaussians)
    #
    # bulge_gaussian_list = []
    #
    # for j in range(gaussian_per_basis):
    #     gaussian_list = af.Collection(
    #         af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    #     )
    #
    #     for i, gaussian in enumerate(gaussian_list):
    #         gaussian.centre.centre_0 = centre_0
    #         gaussian.centre.centre_1 = centre_1
    #         gaussian.ell_comps = gaussian_list[0].ell_comps
    #         gaussian.sigma = 10 ** log10_sigma_list[i]
    #
    #     bulge_gaussian_list += gaussian_list
    #
    # source_bulge = af.Model(
    #     al.lp_basis.Basis,
    #     profile_list=bulge_gaussian_list,
    # )
    #
    # source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)
    #
    # # Overall Lens Model:
    #
    # model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    # n_dim = 17

    analysis = al.AnalysisImaging(
        dataset=dataset,
    )

    from autofit.non_linear.fitness import Fitness

    class FitnessActor(Fitness):
        pass

    # 1) Start your client
    client = Client(n_workers=4, threads_per_worker=1, processes=True)
    print("Dask dashboard:", client.dashboard_link)

    # 1. Submit all ActorFutures first
    actors = [
        client.submit(
            FitnessActor, model, analysis, None, True, -1.0e99, False, False, actor=True
        ).result()
        for _ in range(4)
    ]

    # 2. Wait for all actors to initialize and unwrap them to proxies
    #    actors = [afut.result() for afut in actor_futures]

    # 3) Generate some test points
    n_batch = 100

    points = np.random.rand(n_batch, n_dim).astype(np.float64)
    points = af.PriorVectorized(model)((points))

    import time

    start = time.time()

    results = []
    for i, pt in enumerate(points):
        actor = actors[i % len(actors)]
        fut = actor.call_numpy_wrapper(pt)  # This is a Dask Future
        val = fut.result()  # BLOCK until that Future yields a float

        results.append(val)

    print(results)
    print(f"Time taken serial: {time.time() - start:.2f} seconds")

    import time

    start = time.time()

    results = []
    for i, pt in enumerate(points):
        actor = actors[i % len(actors)]
        fut = actor.call_numpy_wrapper(pt)  # This is a Dask Future
        val = fut.result()  # BLOCK until that Future yields a float

        results.append(val)

    print(results)
    print(f"Time taken serial: {time.time() - start:.2f} seconds")

    start = time.time()

    # Submit all 100 jobs first
    futures = []
    for i, pt in enumerate(points):
        actor = actors[i % len(actors)]
        futures.append(actor.call_numpy_wrapper(pt))  # returns ActorFuture

    print(f"Time to submit jobs: {time.time() - start:.2f} seconds")

    start = time.time()

    # # Submit all 100 jobs first
    # futures = []
    # for i, pt in enumerate(points):
    #     actor = actors[i % len(actors)]
    #     futures.append(actor.call_numpy_wrapper(pt))  # returns ActorFuture
    #
    # # Wait for all 100 to finish together
    # results = [f.result() for f in futures]
    #
    # print(results[0])
    # print(f"Time taken parallel: {time.time() - start:.2f} seconds")

    start = time.time()

    # Wait for all 100 to finish together
    from itertools import cycle

    actor_cycle = cycle(actors)
    futures = [actor.call(pt) for actor, pt in zip(actor_cycle, points)]
    results = [f.result() for f in futures]

    print(results)
    print(f"Time taken parallel: {time.time() - start:.2f} seconds")

    client.close()


if __name__ == "__main__":
    main()


# SINGLE CPU:
# Time taken for single CPU: 11.3490 seconds
