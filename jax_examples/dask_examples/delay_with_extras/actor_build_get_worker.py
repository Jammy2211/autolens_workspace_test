def build_analysis():

    from os import path

    import autolens as al

    dataset_name = "source_complex"
    dataset_path = path.join("dataset", "imaging", dataset_name)

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=0.05,
    )

    mask_2d = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.5
    )

    dataset = dataset.apply_mask(mask=mask_2d)

    dataset = dataset.apply_over_sampling(over_sample_size_lp=8)

    dataset.convolver

    return al.AnalysisImaging(
        dataset=dataset,
    )


# -------------------------
# Helpers to build on worker
# -------------------------
def build_model():

    import numpy as np

    import autofit as af
    import autolens as al

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

    return model


from dask.distributed import get_worker


class FitnessActor:
    """
    Lightweight actor. Build model & analysis on the worker with build_resources,
    then create the real Fitness instance on the worker for fast repeated calls.
    """

    def __init__(self):

        # keep it light
        self.fitness = None
        self._built = False

    def build_resources(self):
        """
        Run on worker: build model & analysis and create the actual Fitness object.
        Optionally precompile JAX for the shape (batch_size, n_dim) if provided.
        """

        worker = get_worker()

        if not hasattr(worker, "analysis"):
            worker.analysis = build_analysis()  # Worker builds once
        if not hasattr(worker, "model"):
            worker.model = build_model()

        if not hasattr(worker, "fitness"):

            # Import the Fitness class here and construct it on the worker
            from autofit.non_linear.fitness import Fitness as AF_Fitness

            # Create the Fitness instance *on the worker* (no large transfer)
            # Adjust constructor args to match your actual signature if different
            worker.fitness = AF_Fitness(
                worker.model, worker.analysis, None, True, -1.0e99, False, False
            )

    def call_numpy_wrapper(self, pt):
        """
        Process a batch of points in one go.
        pts_batch: shape (batch_size, n_dim)
        """

        self.build_resources()

        worker = get_worker()

        # instance = worker.model.instance_from_vector(pt)
        #
        # return worker.analysis.log_likelihood_function(instance=instance)

        return worker.fitness.call_numpy_wrapper(pt)


def main():

    import time as t
    from dask.distributed import Client
    import numpy as np

    n_batch = 20
    n_dim = 19

    time_dict = {}

    for n_workers in [1, 4]:

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        print("Dashboard:", client.dashboard_link)

        actors = [
            client.submit(FitnessActor, actor=True).result() for _ in range(n_workers)
        ]

        futures = []

        #        points = np.random.rand(n_batch, n_dim)
        points = np.load("points.npy")

        for i, pt in enumerate(points):
            actor = actors[i % len(actors)]
            futures.append(actor.call_numpy_wrapper(pt))  # returns ActorFuture

        # Wait for all 100 to finish together
        results = [f.result() for f in futures]

        start_time = t.time()

        print("STARTING")
        print()
        print()
        print()

        for i in range(3):

            points = np.load("points.npy")
            points = np.asarray(points, dtype=np.float64)

            #            points = np.asarray(np.random.rand(n_batch, n_dim), dtype=np.float64)

            futures = []

            for i, pt in enumerate(points):
                actor = actors[i % len(actors)]
                futures.append(actor.call_numpy_wrapper(pt))  # returns ActorFuture

            results = [f.result() for f in futures]

        end_time = t.time()

        print()
        print()
        print()
        print("Results:", results)
        print(
            f"Time: for n workers = {n_workers} time = {end_time - start_time:.2f} sec"
        )

        time_dict[n_workers] = end_time - start_time

        client.close()

    print(time_dict)


if __name__ == "__main__":
    main()
