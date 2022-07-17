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
 - The lens galaxy's dark matter mass distribution is modeled as a `EllNFWMCRLudlow`.
 - The source galaxy's light is a parametric `Inversion`.

This runner uses the SLaM pipelines:

 `source_parametric/with_lens_light`
 `source_pixelized/with_lens_light`
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
    path_prefix=path.join("slam", "rjlens"), number_of_cores=1, session=None
)

"""
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit.
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

 - Uses a parametric `EllSersic` bulge and `EllExponential` disk with centres aligned for the lens
 galaxy's light.

 - Uses an `EllIsothermal` model for the lens's total mass distribution with an `ExternalShear`.
"""
analysis = al.AnalysisImaging(dataset=imaging)

source_parametric_results = slam.source_parametric.no_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    mass=af.Model(al.mp.EllIsothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.EllSersic),
    redshift_lens=0.169,
    redshift_source=0.451,
)

"""
__SOURCE PIXELIZED PIPELINE (with lens light)__

The SOURCE PIXELIZED PIPELINE (with lens light) uses four searches to initialize a robust model for the `Inversion` 
that reconstructs the source galaxy's light. It begins by fitting a `VoronoiMagnification` pixelization with `Constant` 
regularization, to set up the model and hyper images, and then:

 - Uses a `VoronoiBrightnessImage` pixelization.

 - Uses an `AdaptiveBrightness` regularization.

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PARAMETRIC PIPELINE through to the
 SOURCE PIXELIZED PIPELINE.
"""
analysis = al.AnalysisImaging(dataset=imaging)

source_pixelized_results = slam.source_pixelized.no_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_parametric_results=source_parametric_results,
    pixelization=al.pix.VoronoiBrightnessImage,
    regularization=al.reg.AdaptiveBrightness,
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

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIXELIZED PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PARAMETRIC PIPELINE through to the MASS 
 LIGHT DARK PIPELINE.

__Settings__:

 - Hyper: We may be using hyper features and therefore pass the result of the SOURCE PIXELIZED PIPELINE to use as the
 hyper dataset if required.

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_inversion.py`) to remove unphysical solutions from the `Inversion` model-fitting.

"""
analysis = al.AnalysisImaging(
    dataset=imaging, hyper_dataset_result=source_pixelized_results.last
)

lens_bulge = af.Model(al.mp.EllSersic)
lens_disk = af.Model(al.mp.EllSersic)

lens_bulge.centre_0 = 0.0033909063521568973
lens_bulge.centre_1 = -0.009346694405826354
lens_bulge.elliptical_comps.elliptical_comps_0 = 0.0309506766816977
lens_bulge.elliptical_comps.elliptical_comps_1 = -0.0621964387481318
lens_bulge.intensity = 0.22305566711518723
lens_bulge.effective_radius = 0.4624170312787752
lens_bulge.sersic_index = 1.2671926982808182

lens_disk.centre_0 = 0.033660227654696305
lens_disk.centre_1 = 0.06862843744615527
lens_disk.elliptical_comps.elliptical_comps_0 = 0.16208666776894873
lens_disk.elliptical_comps.elliptical_comps_1 = -0.14256415128388042
lens_disk.intensity = 0.02901283096677522
lens_disk.effective_radius = 5.131096088458944
lens_disk.sersic_index = 1.3062257408698343

dark = af.Model(al.mp.EllNFWMCRLudlow)

mass_results = slam.mass_light_dark.no_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_pixelized_results,
    lens_bulge=lens_bulge,
    lens_disk=lens_disk,
    lens_envelope=None,
    dark=dark,
)


import sys

sys.exit()

"""
Finish.
"""
