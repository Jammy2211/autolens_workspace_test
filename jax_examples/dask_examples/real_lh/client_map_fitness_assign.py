from autoconf import cached_property
import time as t
import jax
import jax.numpy as jnp
from dask.distributed import Client
import numpy as np
from os import path
from autofit.non_linear.fitness import Fitness

import autofit as af
import autolens as al


# Module-level helper so dask can pickle it easily
def process_fitness(params: np.ndarray, fitness_obj: Fitness) -> float:
    """Run the fitness call. fitness_obj is expected to be the scattered object on the worker."""
    return float(fitness_obj(params))


def _compile_on_worker(fitness_obj: Fitness) -> bool:
    """
    Force the fitness object's `.single_jit` property to be accessed on the worker,
    which triggers compilation there. Returns True when done.
    """
    _ = fitness_obj.call_numpy_wrapper
    return True


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

    analysis = al.AnalysisImaging(
        dataset=dataset,
    )

    n_batch = 20
    n_dim = 19

    time_dict = {}

    for n_workers in [1, 2, 4, 8]:

        # Use processes=True so each worker has its own process / JAX compilation cache.
        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        print("Started client:", client)

        # Identify workers' addresses
        worker_addrs = list(client.scheduler_info()["workers"].keys())
        print("Workers:", worker_addrs)

        # Create and scatter one Fitness instance to each worker (not broadcast)
        per_worker_futures = {}
        for addr in worker_addrs:
            # instantiate on driver but scatter to specific worker
            fitness = Fitness(
                model=model,
                analysis=analysis,
                fom_is_log_likelihood=True,
                resample_figure_of_merit=-1.0e99,
            )
            future = client.scatter(fitness, workers=[addr], broadcast=False)
            per_worker_futures[addr] = future

        points = np.random.rand(n_batch, n_dim).astype(np.float64)

        futures = []
        for i, pt in enumerate(points):
            addr = worker_addrs[i % len(worker_addrs)]
            # schedule process_fitness on the worker that holds the Fitness object
            futures.append(
                client.submit(
                    process_fitness, pt, per_worker_futures[addr], workers=[addr]
                )
            )
        results = client.gather(futures)

        print("Pre-compilation done on all workers.")

        print("STARTING")
        print()
        print()
        print()

        # Test run (measure timings)
        start = t.time()

        for i in range(50):

            points = np.random.rand(n_batch, n_dim).astype(np.float64)

            futures = []
            for i, pt in enumerate(points):
                addr = worker_addrs[i % len(worker_addrs)]
                # schedule process_fitness on the worker that holds the Fitness object
                futures.append(
                    client.submit(
                        process_fitness, pt, per_worker_futures[addr], workers=[addr]
                    )
                )

            results = client.gather(futures)

        print(results)
        end = t.time()
        print()
        print()
        print()
        print(f"Time for {n_workers} workers: {end - start:.2f} sec")

        client.close()

    print(time_dict)


if __name__ == "__main__":
    main()
