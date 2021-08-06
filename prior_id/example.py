import os
from os import path
import sys
import json

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "slacs"))

conf.instance["general"]["remove_files"] = False

""" 
__AUTOLENS + DATA__
"""
import autofit as af
import autolens as al

import os

sys.path.insert(0, os.getcwd())
import slam_prior_id

pixel_scales = 0.05

dataset_name = "slacs0252+0039"

dataset_path = path.join("prior_id")

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image_scaled_lens_sub_padded.fits",
    psf_path=f"{dataset_path}/F814W_psf.fits",
    noise_map_path=f"{dataset_path}/noise_map_scaled.fits",
    pixel_scales=pixel_scales,
    name=dataset_name,
)

mask = al.Mask2D.circular_annular(
    shape_native=imaging.shape_native,
    pixel_scales=pixel_scales,
    centre=(0.0, 0.0),
    inner_radius=0.25,
    outer_radius=1.6,
)

imaging = imaging.apply_mask(mask=mask)

imaging = imaging.apply_settings(
    settings=al.SettingsImaging(grid_class=al.Grid2DIterate, fractional_accuracy=0.9999)
)


"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, databse use, etc.
"""
settings_autofit = slam_prior_id.SettingsAutoFit(
    path_prefix="slacs", unique_tag=dataset_name, number_of_cores=1, session=None
)

"""
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit.
"""
setup_hyper = al.SetupHyper(
    hyper_galaxies_lens=True,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
    search_dict={"nlive": 30, "sample": "rwalk"},
)

"""
__SOURCE PARAMETRIC PIPELINE (with lens light)__

The SOURCE PARAMETRIC PIPELINE (with lens light) uses three searches to initialize a robust model for the 
source galaxy's light, which in this example:

 - Uses a parametric `EllSersic` bulge and `EllExponential` disk with centres aligned for the lens
 galaxy's light.

 - Uses an `EllIsothermal` model for the lens's total mass distribution with an `ExternalShear`.

 __Settings__:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
"""
mass = af.Model(al.mp.EllIsothermal)

analysis = al.AnalysisImaging(dataset=imaging)

bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllExponential)
bulge.centre = disk.centre

source_parametric_results = slam_prior_id.source_parametric.with_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    lens_bulge=bulge,
    lens_disk=disk,
    mass=mass,
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.EllSersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=0.28,
    redshift_source=0.982,
)

"""
__SOURCE INVERSION PIPELINE (with lens light)__

The SOURCE INVERSION PIPELINE (with lens light) uses four searches to initialize a robust model for the `Inversion` 
that reconstructs the source galaxy's light. It begins by fitting a `VoronoiMagnification` pixelization with `Constant` 
regularization, to set up the model and hyper images, and then:

 - Uses a `VoronoiBrightnessImage` pixelization.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PARAMETRIC PIPELINE through to the
 SOURCE INVERSION PIPELINE.
"""
analysis = al.AnalysisImaging(
    dataset=imaging, hyper_dataset_result=source_parametric_results.last
)

source_inversion_results = slam_prior_id.source_inversion.with_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_parametric_results=source_parametric_results,
    pixelization=al.pix.VoronoiBrightnessImage,
    regularization=al.reg.AdaptiveBrightness,
)

"""
__LIGHT PARAMETRIC PIPELINE__

The LIGHT PARAMETRIC PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE PARAMETRIC PIPELINE.
In this example it:

 - Uses a parametric `EllSersic` bulge and `EllSersic` disk with centres aligned for the lens galaxy's 
 light [Do not use the results of the SOURCE PARAMETRIC PIPELINE to initialize priors].

 - Uses an `EllIsothermal` model for the lens's total mass distribution [fixed from SOURCE PARAMETRIC PIPELINE].

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE INVERSION PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values].

__Preloads__: 

 - Inversion: We preload linear algebra matrices used by the inversion using the maximum likelihood hyper-result of the 
 SOURCE INVERSION PIPELINE. This ensures these matrices are not recalculated every iteration of the log likelihood 
 function, speeding up the model-fit (this is possible because the mass model and source pixelization are fixed).   
"""
preloads = al.Preloads.setup(result=source_inversion_results.last.hyper, inversion=True)

analysis = al.AnalysisImaging(
    dataset=imaging,
    hyper_dataset_result=source_inversion_results.last,
    preloads=preloads,
)

bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllSersic)
disk.centre = bulge.centre
bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

light_results = slam_prior_id.light_parametric.with_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_inversion_results,
    lens_bulge=bulge,
    lens_disk=disk,
    end_with_hyper_extension=True,
)


"""
__MASS TOTAL PIPELINE (with lens light)__

The MASS TOTAL PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE INVERSION PIPELINE to initialize the model priors and the lens 
light model of the LIGHT PARAMETRIC PIPELINE. In this example it:

 - Uses a parametric `EllSersic` bulge and `EllSersic` disk with centres aligned for the lens galaxy's 
 light [fixed from LIGHT PARAMETRIC PIPELINE].

 - Uses an `EllPowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE 
 PARAMETRIC PIPELINE + centre unfixed from (0.0, 0.0)].

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE INVERSION PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS TOTAL 
 PIPELINE.

__Settings__:

 - Hyper: We may be using hyper features and therefore pass the result of the SOURCE INVERSION PIPELINE to use as the
 hyper dataset if required.

__Preloads__:

 - Pixelization: We preload the pixelization using the maximum likelihood hyper-result of the SOURCE INVERSION PIPELINE. 
 This ensures the source pixel-grid is not recalculated every iteration of the log likelihood function, speeding up 
 the model-fit (this is only possible because the source pixelization is fixed). 
"""
preloads = al.Preloads.setup(result=light_results.last.hyper, pixelization=True)

analysis = al.AnalysisImaging(
    dataset=imaging,
    hyper_dataset_result=light_results.last,
    settings_lens=settings_lens,
    preloads=preloads,
)

mass_results = slam_prior_id.mass_total.with_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_inversion_results,
    light_results=light_results,
    mass=af.Model(al.mp.EllPowerLaw),
    end_with_hyper_extension=True,
)

"""
Finish.
"""
