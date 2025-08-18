from autoconf import cached_property
import jax
from dask.distributed import Client
from os import path

import autofit as af
import autolens as al

import time
import numpy as np

from jax import debug


class Fitness:

    def __init__(self, delay=0.2, analysis=None, model=None):

        self.delay = delay
        self.analysis = analysis
        self.model = model

    def call_numpy_wrapper(self, parameters):

        figure_of_merit = self.call_jit(np.array(parameters))

        return figure_of_merit.item()

    @cached_property
    def call_jit(self):
        debug.print("Compiling fitness function for JAX...")
        return jax.jit(self.call)

    def call(self, params):

        time.sleep(self.delay)

        return 1.0


#        instance = self.model.instance_from_vector(vector=params)

# Evaluate log likelihood (must be side-effect free and exception-free)
#        return self.analysis.log_likelihood_function(instance=instance)


class FitnessActor(Fitness):
    pass


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

    for n_workers in [1, 4]:

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        print("Dashboard:", client.dashboard_link)

        actors = [
            client.submit(FitnessActor, 0.1, analysis, model, actor=True).result()
            for _ in range(n_workers)
        ]

        """
        WARM UP JAX JIT
        """
        points = np.random.rand(n_batch, n_dim)

        futures = []

        for i, pt in enumerate(points):
            actor = actors[i % len(actors)]
            futures.append(actor.call_numpy_wrapper(pt))  # returns ActorFuture

        points = np.random.rand(n_batch, n_dim)

        """
        NOW TIME THE DASK SPEED UP
        """
        import time as t

        start_time = t.time()

        # Submit all 100 jobs first
        futures = []

        for i, pt in enumerate(points):
            actor = actors[i % len(actors)]
            futures.append(actor.call_numpy_wrapper(pt))  # returns ActorFuture

        # Wait for all 100 to finish together
        results = [f.result() for f in futures]

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
