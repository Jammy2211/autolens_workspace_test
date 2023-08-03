"""
Basis Prior Passing Instance Test
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from typing import Tuple, Optional, Union
import numpy as np
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "slam"))

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.2,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_autofit = af.SettingsSearch(
    path_prefix=path.join("basis_prior_passing", "instance"),
    number_of_cores=1,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).
"""
redshift_lens = 0.5
redshift_source = 1.0


"""
__Search 1__
"""
mass = af.Model(al.mp.Isothermal)

total_gaussians = 30
gaussian_per_basis = 2

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
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

basis = af.Model(
    al.lp_basis.Basis,
    light_profile_list=bulge_gaussian_list,
    regularization=al.reg.ConstantZeroth(
        coefficient_neighbor=0.0, coefficient_zeroth=1.0
    ),
)

model_1 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=redshift_lens, mass=mass),
        source=af.Model(
            al.Galaxy,
            redshift=redshift_source,
            bulge=basis,
        ),
    ),
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

search_1 = af.DynestyStatic(
    name="source_lp_cast",
    **settings_autofit.search_dict,
    nlive=200,
    walks=10,
)

result_1 = search_1.fit(model=model_1, analysis=analysis_1, **settings_autofit.fit_dict)


def cast_method(
    component: Union[al.LightProfile], fit: Union[al.FitImaging, al.FitInterferometer]
) -> al.lp_basis.Basis:
    light_profile_list = []

    for light_profile in component.light_profile_list:
        intensity = fit.linear_light_profile_intensity_dict[light_profile]

        if isinstance(light_profile, al.lp_linear.LightProfileLinear):
            light_profile_list.append(
                light_profile.lp_instance_from(intensity=intensity)
            )

        else:
            light_profile_list.append(light_profile)

    basis = al.lp_basis.Basis(light_profile_list=light_profile_list)

    return basis


# The cast method above works on an input light profile component, e.g. the `Basis` class.

# The solution should allow us to cast everything at once via a single function.

basis_casted = cast_method(
    component=result_1.instance.galaxies.source.bulge,
    fit=result_1.max_log_likelihood_fit,
)

basis_casted = af.Model(basis_casted)

print(basis_casted.info)

# cast_model(
#     model=result.model.galaxies.lens.bulge, # This contains all the Gaussians
#     model_parameter_dict=linear_light_profile_intensity_dict,  # The dictionary containing the values of parameters to cast.
#     parameter_name="intensity",  # The parameter whose values are cast and updated.
#     model_cast_from, al.lp_linear.Gaussian, # Only cast model components of type al.lp_linear.Gaussian
#     model_to_cast=al.lp.Gaussian # Only cast from al.lp_linear.Gaussian to al.Gaussian
# )
