from autoconf import cached_property
import time as t
import jax
import jax.numpy as jn
from dask.distributed import Client
import numpy as np
from os import path

import autofit as af
import autolens as al


import time
from dask.distributed import Client
import numpy as np


class Fitness:

    def __init__(self, delay=0.2, analysis=None, model=None):
        self.delay = delay
        self.analysis = analysis
        self.model = model

    def call(self, params):

        instance = self.model.instance_from_vector(vector=params)

        # Evaluate log likelihood (must be side-effect free and exception-free)
        return self.analysis.log_likelihood_function(instance=instance)


# 2) Process‐fitness is also at module scope (so dask can pickle it easily)
def process_fitness(params: np.ndarray, fitness_future) -> float:
    # convert incoming numpy→device array, call the jitted fn,
    # and return a Python float
    time.sleep(0.2)
    fitnss = fitness_future
    return fitnss.call(np.array(params))


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
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=5.0
    )

    dataset = dataset.apply_mask(mask=mask_2d)

    dataset = dataset.apply_over_sampling(over_sample_size_lp=8)

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

    for n_workers in [1, 4]:

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        print("Dashboard:", client.dashboard_link)

        fitness = Fitness(analysis=analysis, model=model)

        func = client.scatter(fitness, broadcast=True)

        points = np.random.rand(n_batch, n_dim)

        import time as t

        start_time = t.time()

        # client.map can take multiple iterables; we repeat the jitted_future
        futures = client.map(
            process_fitness,
            points,  # each element is a (19,) array
            [func] * n_batch,  # broadcast the same future
        )
        results = client.gather(futures)

        end_time = t.time()

        print("Results:", results)
        print(
            f"Time: for n workers = {n_workers} time = {end_time - start_time:.2f} sec"
        )

        time_dict[n_workers] = end_time - start_time

        client.close()

    print(time_dict)


if __name__ == "__main__":
    main()
