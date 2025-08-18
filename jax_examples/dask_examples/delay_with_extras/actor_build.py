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

    dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

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

    total_gaussians = 30
    gaussian_per_basis = 2

    # The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
    mask_radius = 3.0
    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    # By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.

    centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        # A list of Gaussian model components whose parameters are customized belows.

        gaussian_list = af.Collection(
            af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
        )

        # Iterate over every Gaussian and customize its parameters.

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
            gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
            gaussian.ell_comps = gaussian_list[
                0
            ].ell_comps  # All Gaussians have same elliptical components.
            gaussian.sigma = (
                10 ** log10_sigma_list[i]
            )  # All Gaussian sigmas are fixed to values above.

        bulge_gaussian_list += gaussian_list

    # The Basis object groups many light profiles together into a single model component.

    bulge = af.Model(
        al.lp_basis.Basis,
        profile_list=bulge_gaussian_list,
    )

    mass = af.Model(al.mp.Isothermal)

    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

    # Source:

    total_gaussians = 20
    gaussian_per_basis = 1

    # By defining the centre here, it creates two free parameters that are assigned to the source Gaussians.

    centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    log10_sigma_list = np.linspace(-2, np.log10(1.0), total_gaussians)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        gaussian_list = af.Collection(
            af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    source_bulge = af.Model(
        al.lp_basis.Basis,
        profile_list=bulge_gaussian_list,
    )

    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

    # Overall Lens Model:

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    return model


# class Fitness:
#     def __init__(self, delay=0.2, analysis=None, model=None):
#         self.delay = delay
#         self.analysis = analysis
#         self.model = model
#
#     def call(self, params):
#
#         instance = self.model.instance_from_vector(vector=params)
#
#         # Evaluate log likelihood (must be side-effect free and exception-free)
#         return self.analysis.log_likelihood_function(instance=instance)
#
#
#
# class FitnessActor(Fitness):
#     pass


class FitnessActor:
    """
    Lightweight actor. Build model & analysis on the worker with build_resources,
    then create the real Fitness instance on the worker for fast repeated calls.
    """

    def __init__(self, delay=0.2):

        # keep it light
        self.delay = delay
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

    def call_numpy_wrapper(self, pt):
        """
        Process a batch of points in one go.
        pts_batch: shape (batch_size, n_dim)
        """

        self.build_resources()

        return self.fitness.call_numpy_wrapper(pt)

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

    n_batch = 20
    n_dim = 17

    batch_size = 25

    time_dict = {}

    for n_workers in [1, 4]:

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        print("Dashboard:", client.dashboard_link)

        actors = [
            client.submit(FitnessActor, 0.05, actor=True).result()
            for _ in range(n_workers)
        ]

        futures = []

        points = np.random.rand(n_batch, n_dim)

        for i, pt in enumerate(points):
            actor = actors[i % len(actors)]
            futures.append(actor.call_numpy_wrapper(pt))  # returns ActorFuture

        # Wait for all 100 to finish together
        results = [f.result() for f in futures]

        # for start in range(0, n_batch, batch_size):
        #     batch = points[start:start + batch_size]
        #     actor = actors[(start // batch_size) % len(actors)]
        #     futures.append(actor.call_numpy_wrapper_batch(batch))
        #
        # results_nested = [f.result() for f in futures]
        # results = [item for sublist in results_nested for item in sublist]

        start_time = t.time()

        print("STARTING")
        print()
        print()
        print()

        for i in range(3):

            points = np.asarray(np.random.rand(n_batch, n_dim), dtype=np.float64)

            futures = []

            for i, pt in enumerate(points):
                actor = actors[i % len(actors)]
                futures.append(actor.call_numpy_wrapper(pt))  # returns ActorFuture

            results = [f.result() for f in futures]

            # for start in range(0, n_batch, batch_size):
            #     batch = points[start:start + batch_size]
            #     actor = actors[(start // batch_size) % len(actors)]
            #     futures.append(actor.call_numpy_wrapper_batch(batch))
            #
            # results_nested = [f.result() for f in futures]
            # results = [item for sublist in results_nested for item in sublist]

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
