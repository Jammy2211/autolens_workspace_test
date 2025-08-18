def build_analysis():

    from os import path

    import autolens as al

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

    return al.AnalysisImaging(
        dataset=dataset,
    )


# -------------------------
# Helpers to build on worker
# -------------------------
def build_model():

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

        if self._built:
            return

        # Build heavy objects here (on worker process)
        model = build_model()
        analysis = build_analysis()

        # Import the Fitness class here and construct it on the worker
        from autofit.non_linear.fitness import Fitness as AF_Fitness

        # Create the Fitness instance *on the worker* (no large transfer)
        # Adjust constructor args to match your actual signature if different
        self.fitness = AF_Fitness(model, analysis, None, True, -1.0e99, False, False)

        self._built = True

    def call_numpy_wrapper_batch(self, pts_batch):
        """
        Process a batch of points in one go.
        pts_batch: shape (batch_size, n_dim)
        """

        self.build_resources()

        results = []
        for pt in pts_batch:
            results.append(self.fitness.call_numpy_wrapper(pt))
        return results


def main():

    import time as t
    from dask.distributed import Client
    import numpy as np

    n_batch = 100
    n_dim = 19
    batch_size = 25

    time_dict = {}

    for n_workers in [1, 4]:

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        print("Dashboard:", client.dashboard_link)

        # 1. Submit all ActorFutures first
        actors = [
            client.submit(FitnessActor, actor=True).result() for _ in range(n_workers)
        ]

        # Submit all 100 jobs first
        futures = []

        points = np.random.rand(n_batch, n_dim)

        for start in range(0, n_batch, batch_size):
            batch = points[start : start + batch_size]
            actor = actors[(start // batch_size) % len(actors)]
            futures.append(actor.call_numpy_wrapper_batch(batch))

        results_nested = [f.result() for f in futures]
        results = [item for sublist in results_nested for item in sublist]

        start_time = t.time()

        print("STARTING")
        print()
        print()
        print()

        for i in range(3):

            points = np.asarray(np.random.rand(n_batch, n_dim), dtype=np.float64)

            # Submit all 100 jobs first
            futures = []

            for start in range(0, n_batch, batch_size):
                batch = points[start : start + batch_size]
                actor = actors[(start // batch_size) % len(actors)]
                futures.append(actor.call_numpy_wrapper_batch(batch))

            results_nested = [f.result() for f in futures]
            results = [item for sublist in results_nested for item in sublist]

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
