"""
SLaM (Source, Light and Mass): Light Parametric + Mass Total + Source Parametric
================================================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `EllSersic` profile for the bulge, this will be used in the subsequent MASS LIGHT DARK PIPELINE.

Using a SOURCE PARAMETRIC PIPELINE, LIGHT PIPELINE and a MASS LIGHT DARK PIPELINE this SLaM script fits `Imaging` of
a strong lens system, where in the final model:

 - The lens galaxy's light is a bulge+disk `EllSersic` and `EllSersic`.
 - The lens galaxy's stellar mass distribution is a bulge+disk tied to the light model above.
 - The source galaxy's light is a parametric `EllSersic`.

This runner uses the SLaM pipelines:

 `source_parametric/source_parametric__with_lens_light`
 `light_parametric/with_lens_light`
 `mass_total/mass_light_dark`

Check them out for a detailed description of the analysis!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

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

masked_imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(
    imaging=masked_imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join(
    "imaging",
    "slam",
    "light_sersic__mass_light_dark__subhalo_nfw__source_parametric",
    "sensitivity",
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
of different models in the LIGHT PIPELINE and MASS LIGHT DARK PIPELINE can be performed consistently.
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

 - Uses a parametric `EllSersic` bulge and `EllSersic` disk with centres aligned for the lens
 galaxy's light.

 - Uses an `EllIsothermal` model for the lens's total mass distribution with an `ExternalShear`.

 Settings:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS LIGHT DARK 
 PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=masked_imaging)

bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllSersic)
bulge.centre = (0.0, 0.0)
disk.centre = (0.0, 0.0)

source_parametric_results = slam.source_parametric.with_lens_light(
    path_prefix=path_prefix,
    unique_tag=dataset_name,
    analysis=analysis,
    setup_hyper=setup_hyper,
    lens_bulge=bulge,
    lens_disk=disk,
    mass=af.Model(al.mp.EllIsothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.EllSersic),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__LIGHT PARAMETRIC PIPELINE__

The LIGHT PARAMETRIC PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE PARAMETRIC PIPELINE.
In this example it:

 - Uses a parametric `EllSersic` bulge and `EllSersic` disk with centres aligned for the lens galaxy's 
 light [Do not use the results of the SOURCE PARAMETRIC PIPELINE to initialize priors].

 - Uses an `EllIsothermal` model for the lens's total mass distribution [fixed from SOURCE PARAMETRIC PIPELINE].

 - Uses the `EllSersic` model representing a bulge for the source's light [fixed from SOURCE PARAMETRIC PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values].
"""
bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllSersic)
bulge.centre = disk.centre

light_results = slam.light_parametric.with_lens_light(
    path_prefix=path_prefix,
    unique_tag=dataset_name,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_parametric_results,
    lens_bulge=bulge,
    lens_disk=disk,
)

"""
__MASS LIGHT DARK PIPELINE (with lens light)__

The MASS LIGHT DARK PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of 
accuracy, using the source model of the SOURCE PIPELINE and the lens light model of the LIGHT PARAMETRIC PIPELINE to 
initialize the model priors . In this example it:

 - Uses a parametric `EllSersic` bulge and `EllSersic` disk with centres aligned for the lens galaxy's 
 light and its stellar mass [12 parameters: fixed from LIGHT PARAMETRIC PIPELINE].

 - The lens galaxy's dark matter mass distribution is a `EllNFWMCRLudlow` whose centre is aligned with bulge of 
 the light and stellar mass mdoel above [5 parameters].

 - Uses the `EllSersic` model representing a bulge for the source's light [priors initialized from SOURCE 
 PARAMETRIC PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PARAMETRIC PIPELINE through to the MASS 
 LIGHT DARK PIPELINE.
"""
analysis = al.AnalysisImaging(
    dataset=masked_imaging, hyper_result=source_parametric_results.last
)

mass_results = slam.mass_light_dark.with_lens_light(
    path_prefix=path_prefix,
    unique_tag=dataset_name,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_parametric_results,
    light_results=light_results,
    lens_bulge=af.Model(al.lmp.EllSersic),
    lens_disk=af.Model(al.lmp.EllSersic),
    lens_envelope=None,
    dark=af.Model(al.mp.EllNFWMCRLudlow),
    einstein_mass_range=(0.01, 5.0),
)

"""
__SUBHALO PIPELINE (single plane detection)__

The SUBHALO PIPELINE (single plane detection) consists of the following searches:

 1) Refit the lens and source model, to refine the model evidence for comparing to the models fitted which include a 
 subhalo. This uses the same model as fitted in the MASS PIPELINE. 
 2) Performs a grid-search of non-linear searches to attempt to detect a dark matter subhalo. 
 3) If there is a successful detection a final search is performed to refine its parameters.

For this runner the SUBHALO PIPELINE customizes:

 - The [number_of_steps x number_of_steps] size of the grid-search, as well as the dimensions it spans in arc-seconds.
 - The `number_of_cores` used for the gridsearch, where `number_of_cores > 1` performs the model-fits in paralle using
 the Python multiprocessing module.
"""


class AnalysisImagingSensitivity(al.AnalysisImaging):
    def __init__(self, dataset):

        super().__init__(dataset=dataset)

        self.hyper_galaxy_image_path_dict = (
            mass_results.last.hyper_galaxy_image_path_dict
        )
        self.hyper_model_image = mass_results.last.hyper_model_image


subhalo_results = slam.subhalo.sensitivity_mapping(
    path_prefix=path_prefix,
    analysis_cls=AnalysisImagingSensitivity,
    mask=mask,
    psf=masked_imaging.psf,
    mass_results=mass_results,
    subhalo_mass=af.Model(al.mp.SphNFWMCRLudlow),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
    number_of_cores=2,
)


"""
Finish.
"""
