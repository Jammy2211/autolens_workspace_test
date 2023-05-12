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
dataset_name = "light_sersic__mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
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
imaging_plotter.subplot_dataset()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_autofit = af.SettingsSearch(
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
__Adapt Setup__

The `SetupAdapt` determines which hyper-mode features are used during the model-fit as is used identically to the
hyper pipeline examples.

The `SetupAdapt` input `hyper_fixed_after_source` fixes the hyper-parameters to the values computed by the hyper 
extension at the end of the SOURCE PIPELINE. By fixing the hyper-parameter values at this point, model comparison 
of different models in the LIGHT PIPELINE and MASS LIGHT DARK PIPELINE can be performed consistently.
"""
setup_adapt = al.SetupAdapt()

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
analysis = al.AnalysisImaging(dataset=imaging)

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


source_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
source_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussian_list):

    gaussian.centre = gaussian_list[0].centre
    gaussian.ell_comps = gaussian_list[0].ell_comps
    gaussian.sigma = source_a + (source_b * np.log10(i + 1))

source_bulge = af.Model(al.lp_basis.Basis, light_profile_list=gaussian_list)

source_lp_results = slam.source_lp.run(
    settings_autofit=settings_autofit,
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


light_results = slam.light_lp.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_adapt=setup_adapt,
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
    dataset=imaging, adapt_result=source_lp_results.last
)

mass_results = slam.mass_light_dark.run__from_light_linear(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_results=source_lp_results,
    light_results=light_results,
    dark=af.Model(al.mp.NFWMCRLudlow),
    einstein_mass_range=(0.01, 5.0),
)

"""
Finish.
"""
