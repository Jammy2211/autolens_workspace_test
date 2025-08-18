import jax.numpy as jnp
import jax


class PSFLight:

    def __init__(self, psf):

        self.psf = psf

        self.slim_to_native_tuple = None
        self.slim_to_native_blurring_tuple = None

    def convolve_image(self, image, blurring_image, jax_method="direct"):
        """
        For a given 1D array and blurring array, convolve the two using this psf.

        Parameters
        ----------
        image
            1D array of the values which are to be blurred with the psf's PSF.
        blurring_image
            1D array of the blurring values which blur into the array after PSF convolution.
        jax_method
            If JAX is enabled this keyword will indicate what method is used for the PSF
            convolution. Can be either `direct` to calculate it in real space or `fft`
            to calculated it via a fast Fourier transform. `fft` is typically faster for
            kernels that are more than about 5x5. Default is `fft`.
        """

        slim_to_native_tuple = self.slim_to_native_tuple
        slim_to_native_blurring_tuple = self.slim_to_native_blurring_tuple

        if slim_to_native_tuple is None:

            slim_to_native_tuple = jnp.nonzero(
                jnp.logical_not(image.mask), size=image.shape[0]
            )

        if slim_to_native_blurring_tuple is None:

            slim_to_native_blurring_tuple = jnp.nonzero(
                jnp.logical_not(blurring_image.mask), size=blurring_image.shape[0]
            )

        # make sure dtype matches what you want
        expanded_array_native = jnp.zeros(
            image.mask.shape, dtype=jnp.asarray(image).dtype
        )

        # set using a tuple of index arrays
        expanded_array_native = expanded_array_native.at[slim_to_native_tuple].set(
            jnp.asarray(image)
        )
        expanded_array_native = expanded_array_native.at[
            slim_to_native_blurring_tuple
        ].set(jnp.asarray(blurring_image))

        kernel = self.psf

        convolve_native = jax.scipy.signal.convolve(
            expanded_array_native, kernel, mode="same", method=jax_method
        )

        convolved_array_1d = convolve_native[slim_to_native_tuple]

        return convolved_array_1d


class GridsLight:

    def __init__(self, lp, blurring):

        self.lp = lp
        self.pixelization = None
        self.blurring = blurring
        self.border_relocator = None


class ImagingLight:

    def __init__(self, data, noise_map, psf, grid_lp, grid_blurring):

        self.data = data
        self.noise_map = noise_map
        self.psf = PSFLight(psf)

        self.grids = GridsLight(lp=grid_lp, blurring=grid_blurring)


def build_analysis():

    import numpy as np

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

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge)

    # Overall Lens Model:

    model = af.Collection(galaxies=af.Collection(lens=lens))

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
    n_dim = 6

    time_dict = {}

    for n_workers in [1, 4]:

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        print("Dashboard:", client.dashboard_link)

        actors = [
            client.submit(FitnessActor, actor=True).result() for _ in range(n_workers)
        ]

        futures = []

        points = np.random.rand(n_batch, n_dim)

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

            points = np.asarray(np.random.rand(n_batch, n_dim), dtype=np.float64)

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
