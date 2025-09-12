from autoconf import cached_property
import time as t
import jax
import jax.numpy as jnp
from dask.distributed import Client
import numpy as np
from os import path

import autofit as af
import autolens as al


def process_fitness(params, lookup_fitness) -> float:
    params = np.asarray(params, dtype=np.float64)
    return lookup_fitness.call_numpy_wrapper(params)


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

    from autofit.non_linear.fitness import Fitness

    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    n_batch = 20
    n_dim = 19

    time_dict = {}

    for n_workers in [4]:

        # 3) fire up your client
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

        # 4) scatter the **compiled** function, once:

        jitted_future = client.scatter(fitness, broadcast=True)

        points = np.random.rand(n_batch, n_dim)

        # client.map can take multiple iterables; we repeat the jitted_future
        futures = client.map(
            process_fitness,
            points,  # each element is a (19,) array
            [jitted_future] * n_batch,  # broadcast the same future
        )
        results = client.gather(futures)

        print("STARTING")
        print()
        print()
        print()

        start = t.time()

        for i in range(30):

            points = np.asarray(np.random.rand(n_batch, n_dim), dtype=np.float64)

            #            points = jnp.asarray(np.random.rand(n_batch, n_dim), dtype=jnp.float64)

            # client.map can take multiple iterables; we repeat the jitted_future
            futures = client.map(
                process_fitness,
                points,  # each element is a (19,) array
                [jitted_future] * n_batch,  # broadcast the same future
            )
            results = client.gather(futures)
        #            print(results[0])

        end = t.time()

        print()
        print()
        print()
        print("Results:", results)
        print(f"Time: for n workers = {n_workers} time = {end - start:.2f} sec")

        time_dict[n_workers] = end - start

        client.close()

    print(time_dict)


if __name__ == "__main__":
    main()
