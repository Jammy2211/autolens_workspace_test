"""
SLaM (Source, Light and Mass): Light Parametric + Mass Total + Source Parametric
================================================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS LIGHT DARK PIPELINE.

Using a SOURCE LP PIPELINE, LIGHT PIPELINE and a MASS LIGHT DARK PIPELINE this SLaM script fits `Imaging` of
a strong lens system, where in the final model:

 - The lens galaxy's light is a bulge+disk `Sersic` and `Sersic`.
 - The lens galaxy's stellar mass distribution is a bulge+disk tied to the light model above.
 - The source galaxy's light is a parametric `Sersic`.

This runner uses the SLaM pipelines:

 `source_lp/source_lp__with_lens_light`
 `light_lp`
 `mass_total/mass_light_dark`

Check them out for a detailed description of the analysis!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "slam"))

import autofit as af
import autolens as al
import autolens.plot as aplt
import slam

"""
__Dataset__ 

Load the `Imaging` data, define the `Mask2D` and plot them.
"""
dataset_name = "with_lens_light"
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
settings_search = af.SettingsSearch(
    path_prefix=path.join(
        "slam",
        "source_lp",
        "mass_light_dark",
        "basis",
    ),
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
__SOURCE LP PIPELINE (with lens light)__

The SOURCE LP PIPELINE (with lens light) uses three searches to initialize a robust model for the 
source galaxy's light, which in this example:

 - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens
 galaxy's light.

 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

 Settings:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS LIGHT DARK 
 PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=dataset)

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

lens_bulge = af.Model(
    al.lp_basis.Basis,
    light_profile_list=bulge_gaussian_list,
)


disk_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
disk_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussian_list):
    gaussian.centre = gaussian_list[0].centre
    gaussian.ell_comps = gaussian_list[0].ell_comps
    gaussian.sigma = disk_a + (disk_b * np.log10(i + 1))

lens_disk = af.Model(al.lp_basis.Basis, light_profile_list=gaussian_list)


source_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
source_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussian_list):
    gaussian.centre = gaussian_list[0].centre
    gaussian.ell_comps = gaussian_list[0].ell_comps
    gaussian.sigma = source_a + (source_b * np.log10(i + 1))

source_bulge = af.Model(al.lp_basis.Basis, light_profile_list=gaussian_list)

source_lp_results = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=lens_bulge,
    lens_disk=lens_disk,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE LP PIPELINE.
In this example it:

 - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
 light [Do not use the results of the SOURCE LP PIPELINE to initialize priors].

 - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE LP PIPELINE].

 - Uses the `Sersic` model representing a bulge for the source's light [fixed from SOURCE LP PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values].
"""
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

lens_bulge = af.Model(
    al.lp_basis.Basis,
    light_profile_list=bulge_gaussian_list,
)


disk_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
disk_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussian_list):
    gaussian.centre = gaussian_list[0].centre
    gaussian.ell_comps = gaussian_list[0].ell_comps
    gaussian.sigma = disk_a + (disk_b * np.log10(i + 1))

lens_disk = af.Model(al.lp_basis.Basis, light_profile_list=gaussian_list)


light_results = slam.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_results=source_lp_results,
    lens_bulge=lens_bulge,
    lens_disk=lens_disk,
)

"""
__MASS LIGHT DARK PIPELINE (with lens light)__

The MASS LIGHT DARK PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of 
accuracy, using the source model of the SOURCE PIPELINE and the lens light model of the LIGHT LP PIPELINE to 
initialize the model priors . In this example it:

 - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
 light and its stellar mass [12 parameters: fixed from LIGHT LP PIPELINE].

 - The lens galaxy's dark matter mass distribution is a `NFWMCRLudlow` whose centre is aligned with bulge of 
 the light and stellar mass model above [5 parameters].

 - Uses the `Sersic` model representing a bulge for the source's light [priors initialized from SOURCE 
 PARAMETRIC PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the MASS 
 LIGHT DARK PIPELINE.
"""
analysis = al.AnalysisImaging(
    dataset=dataset, adapt_images=source_lp_results.last.adapt_images
)

mass_results = slam.mass_light_dark.run__from_light_linear(
    settings_search=settings_search,
    analysis=analysis,
    source_results=source_lp_results,
    light_results=light_results,
    dark=af.Model(al.mp.NFWMCRLudlow),
)

"""
Finish.
"""
