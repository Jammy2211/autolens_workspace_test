"""
SLaM (Source, Light and Mass): Light Parametric + Mass Total + Source Parametric
================================================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS PIPELINE.

Using a SOURCE PARAMETRIC PIPELINE, LIGHT PIPELINE and a MASS PIPELINE this SLaM script fits `Imaging` of a strong
lens system, where in the final model:

 - The lens galaxy's light is a bulge+disk `Sersic` and `Exponential`.
 - The lens galaxy's total mass distribution is an `PowerLaw`.
 - The source galaxy's light is a parametric `Sersic`.

This runner uses the SLaM pipelines:

 `source_lp/source_lp__with_lens_light`
 `light_lp`
 `mass_total/mass_total__with_lens_light`

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
dataset_name = "light_sersic__mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)

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
    path_prefix=path.join(
        "slam", "light_sersic__mass_total__source_lp", "linear_light_no_hyper"
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
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit as is used identically to the
hyper pipeline examples.

The `SetupHyper` input `hyper_fixed_after_source` fixes the hyper-parameters to the values computed by the hyper 
extension at the end of the SOURCE PIPELINE. By fixing the hyper-parameter values at this point, model comparison 
of different models in the LIGHT PIPELINE and MASS PIPELINE can be performed consistently.
"""
setup_hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
)

"""
__SOURCE PARAMETRIC PIPELINE (with lens light)__

The SOURCE PARAMETRIC PIPELINE (with lens light) uses three searches to initialize a robust model for the 
source galaxy's light, which in this example:
 
 - Uses a parametric `Sersic` bulge and `Exponential` disk with centres aligned for the lens
 galaxy's light.
 
 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

 Settings:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=imaging)

# max_mge_r = 2.5
# rn = 15
# gaussian_per_basis = 5
#
# log10_sigma_list = np.linspace(-2, np.log10(max_mge_r), rn)
#
# overall_gaussian_list = []
#
# centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
# centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
#
# for j in range(gaussian_per_basis):
#
#     ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
#     ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
#
#     gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(rn))
#
#     for i, gaussian in enumerate(gaussian_list):
#
#         gaussian.centre.centre_0 = centre_0
#         gaussian.centre.centre_1 = centre_1
#         gaussian.ell_comps.ell_comps_0 = ell_comps_0
#         gaussian.ell_comps.ell_comps_1 = ell_comps_1
#         gaussian.sigma = 10 ** log10_sigma_list[i]
#
#     overall_gaussian_list += gaussian_list
#
# lens_bulge = af.Model(al.lp_basis.Basis, light_profile_list=overall_gaussian_list)

bulge_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
bulge_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussian_list):

    gaussian.centre = gaussian_list[0].centre
    gaussian.ell_comps = gaussian_list[0].ell_comps
    gaussian.sigma = bulge_a + (bulge_b * np.log10(i + 1))

lens_bulge = af.Model(al.lp_basis.Basis, light_profile_list=gaussian_list)

disk_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
disk_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussian_list):

    gaussian.centre = gaussian_list[0].centre
    gaussian.ell_comps = gaussian_list[0].ell_comps
    gaussian.sigma = disk_a + (disk_b * np.log10(i + 1))

lens_disk = af.Model(al.lp_basis.Basis, light_profile_list=gaussian_list)

source_bulge = af.Model(al.lp.SersicCore)
source_bulge.radius_break = 0.05
source_bulge.gamma = 0.0
source_bulge.alpha = 2.0

source_lp_results = slam.source_lp.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    lens_bulge=lens_bulge,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp_linear.Sersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE PARAMETRIC PIPELINE.
In this example it:

 - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
 light [Do not use the results of the SOURCE PARAMETRIC PIPELINE to initialize priors].
 
 - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE PARAMETRIC PIPELINE].
 
 - Uses the `Sersic` model representing a bulge for the source's light [fixed from SOURCE PARAMETRIC PIPELINE].
 
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values].
"""
analysis = al.AnalysisImaging(
    dataset=imaging, hyper_dataset_result=source_lp_results.last
)

# max_mge_r = 2.5
# rn = 15
# gaussian_per_basis = 5
#
# log10_sigma_list = np.linspace(-2, np.log10(max_mge_r), rn)
#
# overall_gaussian_list = []
#
# centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
# centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
#
# for j in range(gaussian_per_basis):
#
#     ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
#     ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
#
#     gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(rn))
#
#     for i, gaussian in enumerate(gaussian_list):
#
#         gaussian.centre.centre_0 = centre_0
#         gaussian.centre.centre_1 = centre_1
#         gaussian.ell_comps.ell_comps_0 = ell_comps_0
#         gaussian.ell_comps.ell_comps_1 = ell_comps_1
#         gaussian.sigma = 10 ** log10_sigma_list[i]
#
#     overall_gaussian_list += gaussian_list
#
# lens_bulge = af.Model(al.lp_basis.Basis, light_profile_list=overall_gaussian_list)

bulge_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
bulge_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussian_list):

    gaussian.centre = gaussian_list[0].centre
    gaussian.ell_comps = gaussian_list[0].ell_comps
    gaussian.sigma = bulge_a + (bulge_b * np.log10(i + 1))

lens_bulge = af.Model(al.lp_basis.Basis, light_profile_list=gaussian_list)

disk_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
disk_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussian_list):

    gaussian.centre = gaussian_list[0].centre
    gaussian.ell_comps = gaussian_list[0].ell_comps
    gaussian.sigma = disk_a + (disk_b * np.log10(i + 1))

lens_disk = af.Model(al.lp_basis.Basis, light_profile_list=gaussian_list)

source_bulge = af.Model(al.lp.SersicCore)
source_bulge.radius_break = 0.05
source_bulge.gamma = 0.0
source_bulge.alpha = 2.0

light_results = slam.light_lp.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_lp_results,
    lens_bulge=lens_bulge,
    lens_disk=None,
)

"""
__MASS TOTAL PIPELINE (with lens light)__

The MASS TOTAL PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIPELINE to initialize the model priors and the lens light
model of the LIGHT LP PIPELINE. In this example it:

 - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
 light [fixed from LIGHT LP PIPELINE].

 - Uses an `PowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE 
 PARAMETRIC PIPELINE + centre unfixed from (0.0, 0.0)].
 
 - Uses the `Sersic` model representing a bulge for the source's light [priors initialized from SOURCE 
 PARAMETRIC PIPELINE].
 
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS PIPELINE.
"""
analysis = al.AnalysisImaging(
    dataset=imaging, hyper_dataset_result=source_lp_results.last
)

mass_results = slam.mass_total.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_lp_results,
    light_results=light_results,
    mass=af.Model(al.mp.PowerLaw),
)

"""
Finish.
"""
