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
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.2,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

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

bulge_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
bulge_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussian_list):

    gaussian.centre = gaussian_list[0].centre
    gaussian.ell_comps = gaussian_list[0].ell_comps
    gaussian.sigma = bulge_a + (bulge_b * np.log10(i + 1))

basis = af.Model(al.lp_basis.Basis, light_profile_list=gaussian_list)

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

analysis_1 = al.AnalysisImaging(dataset=imaging)

search_1 = af.DynestyStatic(
    name="source_basis_cast",
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
