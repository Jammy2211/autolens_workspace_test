from os import path

import autofit as af
import autolens as al


import time
from dask.distributed import Client
import numpy as np


def log_likelihood_function(instance, dataset):

    import autolens as al

    galaxies = instance.galaxies

    tracer = al.Tracer(galaxies=galaxies)

    model_image = tracer.blurred_image_2d_from(dataset.grids.lp)

    residual_map = dataset.data - model_image

    log_likelihood = -0.5 * np.sum(
        (residual_map / dataset.noise_map) ** 2
        + np.log(2 * np.pi * dataset.noise_map**2)
    )

    return log_likelihood


class Fitness:
    def __init__(self, delay=0.2, analysis=None, model=None):
        self.delay = delay
        self.analysis = analysis
        self.model = model

    def call(self, params):

        instance = self.model.instance_from_vector(vector=params)

        # Evaluate log likelihood (must be side-effect free and exception-free)
        return self.analysis.log_likelihood_function(instance=instance)


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

    bulge = af.Model(al.lp.Sersic)

    mass = af.Model(al.mp.Isothermal)

    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

    # Source:

    bulge = af.Model(al.lp.Sersic)

    source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

    # Overall Lens Model:

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisImaging(
        dataset=dataset,
    )

    n_batch = 20
    n_dim = 21

    time_dict = {}

    for n_workers in [1, 4]:

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        print("Dashboard:", client.dashboard_link)

        actors = [
            client.submit(FitnessActor, 0.2, analysis, model, actor=True).result()
            for _ in range(n_workers)
        ]

        #        points = np.random.rand(n_batch, n_dim)

        points = np.load("points.npy")

        import time as t

        start_time = t.time()

        # Submit all 100 jobs first
        futures = []

        for i, pt in enumerate(points):
            actor = actors[i % len(actors)]
            futures.append(actor.call(pt))  # returns ActorFuture

        # Wait for all 100 to finish together
        results = [f.result() for f in futures]

        end_time = t.time()

        print("Results:", results)
        print(
            f"Time: for n workers = {n_workers} time = {end_time - start_time:.2f} sec"
        )

        time_dict[n_workers] = end_time - start_time

        client.close()

    print()
    print()
    print()
    print(results)
    print(time_dict)


if __name__ == "__main__":
    main()
