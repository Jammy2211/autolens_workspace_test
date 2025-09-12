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
    return lookup_fitness.call_numpy_wrapper(params)


def main():

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

    n_batch = 100
    n_dim = 17

    time_dict = {}

    for n_workers in [1, 2, 4]:

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

        for i in range(50):

            points = np.asarray(np.random.rand(n_batch, n_dim), dtype=np.float64)

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
