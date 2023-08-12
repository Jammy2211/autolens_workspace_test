"""
Searches: Nautilus
=======================

This example illustrates how to use the nested sampling algorithm Nautilus.

Information about Dynesty can be found at the following links:

 - https://github.com/joshspeagle/dynesty
 - https://dynesty.readthedocs.io/en/latest/
"""
import numpy as np
from os import path
import os
import time

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "searches"))

conf.instance["general"]["model"]["ignore_prior_limits"] = True

import autofit as af
from autofit import exc

def prior_transform(cube, model):

    # `vector_from_unit_vector has a bug which is why we return cube, need to fix.

    return cube

    return model.vector_from_unit_vector(
        unit_vector=cube,
        ignore_prior_limits=True
    )


class Fitness:

    def __init__(
            self, analysis, model,
    ):

        self.analysis = analysis
        self.model = model

    def __call__(self, parameters, *kwargs):

        figure_of_merit = self.figure_of_merit_from(parameter_list=parameters)

        try:
            figure_of_merit = self.figure_of_merit_from(parameter_list=parameters)

            if np.isnan(figure_of_merit):
                return self.resample_figure_of_merit

            return figure_of_merit

        except exc.FitException:
            return self.resample_figure_of_merit

    def fit_instance(self, instance):
        log_likelihood = self.analysis.log_likelihood_function(instance=instance)

        return log_likelihood

    def log_likelihood_from(self, parameter_list):
        instance = self.model.instance_from_vector(vector=parameter_list)
        log_likelihood = self.fit_instance(instance)

        return log_likelihood

    def figure_of_merit_from(self, parameter_list):
        """
        The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space.

        All Nested samplers use the log likelihood.
        """
        return self.log_likelihood_from(parameter_list=parameter_list)

    @property
    def resample_figure_of_merit(self):
        """
        If a sample raises a FitException, this value is returned to signify that the point requires resampling or
        should be given a likelihood so low that it is discard.

        -np.inf is an invalid sample value for Nautilus, so we instead use a large negative number.
        """
        print("RESAMPLED TO NEGATIVE")
        return -1.0e99

def fit():

    """
    __AUTOLENS + DATA__
    """
    import autofit as af
    import autolens as al

    pixel_scales = 0.05

    dataset_name = "slacs0946+1006"

    dataset_path = path.join("naut", dataset_name)

    dataset = al.Imaging.from_fits(
        data_path=f"{dataset_path}/image_lens_light_scaled.fits",
        psf_path=f"{dataset_path}/psf.fits",
        noise_map_path=f"{dataset_path}/noise_map_scaled.fits",
        pixel_scales=pixel_scales,
    )

    mask_radius = 3.5

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=pixel_scales,
        centre=(0.0, 0.0),
        radius=mask_radius
    )

    dataset = dataset.apply_mask(mask=mask)

    dataset = dataset.apply_settings(
        settings=al.SettingsImaging(grid_class=al.Grid2D, sub_size=1)
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
    )

    centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    total_gaussians = 20
    gaussian_per_basis = 1

    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):

        gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians))

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    lens_bulge = af.Model(
        al.lp_basis.Basis,
        light_profile_list=bulge_gaussian_list,
    )

    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    disk_gaussian_list = []

    for j in range(gaussian_per_basis):

        gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians))

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
            gaussian.sigma = 10 ** log10_sigma_list[i]

        disk_gaussian_list += gaussian_list

    lens_disk = af.Model(
        al.lp_basis.Basis,
        light_profile_list=disk_gaussian_list,
    )

    centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

    total_gaussians = 20
    gaussian_per_basis = 1

    log10_sigma_list = np.linspace(-2, np.log10(1.0), total_gaussians)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):

        gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians))

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    source_bulge = af.Model(
        al.lp_basis.Basis,
        light_profile_list=bulge_gaussian_list,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=0.5,
                bulge=lens_bulge,
                disk=lens_disk,
                mass=af.Model(al.mp.Isothermal),
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(
                al.Galaxy,
                redshift=1.0,
                bulge=source_bulge
            ),
        ),
    )

    fitness = Fitness(
        model=model,
        analysis=analysis,
    )

    n_live = 8
    n_dim = model.prior_count

    points = np.zeros((n_live, n_dim))

    for i in range(n_live):

        point = model.random_vector_from_priors_within_limits()
        points[i, :] = point

    from nautilus import Sampler

    number_of_cores_list = [2, 4, 8]

    for number_of_cores in number_of_cores_list:

        sampler = Sampler(
            prior=prior_transform,
            likelihood=fitness.__call__,
            n_dim=model.prior_count,
            prior_kwargs={"model": model},
            pool=number_of_cores,
            n_live=n_live,
        )

        transform = sampler.prior

        args = list(map(transform, np.copy(points)))

        start = time.time()

        time_lh_x1 = 1.0

        if number_of_cores > 1:
            list(sampler.pool_l.map(sampler.likelihood, args))
        else:
            list(map(sampler.likelihood, args))
            time_lh_x1 = time.time() - start

        time_lh = time.time() - start

        print(f"N_CPU: {number_of_cores} / LH Time: {time_lh} / Speed up: {time_lh_x1 / time_lh}")

if __name__ == "__main__":

    fit()