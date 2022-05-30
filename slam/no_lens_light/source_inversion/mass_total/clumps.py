"""
SLaM (Source, Light and Mass): Mass Total + Source Inversion
============================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `EllSersic` profile for the bulge, this will be used in the subsequent MASS PIPELINE.

Using a SOURCE PARAMETRIC PIPELINE, INVERSION SOURCE PIPELINE and a MASS PIPELINE this SLaM script fits `Imaging` of a
strong lens system, where in the final model:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `EllPowerLaw`.
 - The source galaxy is an `Inversion`.

This uses the SLaM pipelines:

 `source__parametric/source_parametric__no_lens_light`
 `source_inversion/source_inversion__no_lens_light`
 `mass__total/mass__total__no_lens_light`

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
    path_prefix=path.join("slam", "mass_total__source_inversion", "no_hyper"),
    number_of_cores=1,
    session=None,
    info={"test": "hello"},
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
    search_pixelized_dict={"maxcall": 3},
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
)

"""
__Clump Model__ 

This model includes clumps, which are `Galaxy` objects with light and mass profiles fixed to an input centre which 
model galaxies nearby the strong lens system.

A full description of the clump API is given in the 
script `autolens_workspace/scripts/imaging/modeling/customize/clumps.py`
"""
clump_centres = al.Grid2DIrregular(grid=[(1.0, 1.0), [2.0, 2.0]])

clump_model = al.ClumpModel(
    redshift=0.5,
    centres=clump_centres,
    mass_cls=al.mp.SphIsothermal,
    einstein_radius_upper_limit=1.0,
)

"""
__SOURCE PARAMETRIC PIPELINE (no lens light)__

The SOURCE PARAMETRIC PIPELINE (no lens light) uses one search to initialize a robust model for the source galaxy's 
light, which in this example:

 - Uses a parametric `EllSersic` bulge for the source's light (omitting a disk / envelope).
 - Uses an `EllIsothermal` model for the lens's total mass distribution with an `ExternalShear`.
 - Fixes the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS PIPELINE).
"""
source_parametric_results = slam.source_parametric.no_lens_light(
    settings_autofit=settings_autofit,
    analysis=al.AnalysisImaging(dataset=imaging),
    setup_hyper=setup_hyper,
    mass=af.Model(al.mp.EllIsothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.EllSersic),
    redshift_lens=0.5,
    redshift_source=1.0,
    clump_model=clump_model,
)

"""
__SOURCE PIXELIZED PIPELINE (no lens light)__

The SOURCE PIXELIZED PIPELINE (no lens light) uses four searches to initialize a robust model for the `Inversion` that
fits the source galaxy's light. It begins by fitting a `VoronoiMagnification` pixelization with `Constant` regularization,
to set up the model and hyper images, and then:

 - Uses a `VoronoiBrightnessImage` pixelization.
 - Uses an `AdaptiveBrightness` regularization.
"""
analysis = al.AnalysisImaging(dataset=imaging)

source_inversion_results = slam.source_inversion.no_lens_light(
    settings_autofit=settings_autofit,
    setup_hyper=setup_hyper,
    analysis=analysis,
    source_parametric_results=source_parametric_results,
    pixelization=al.pix.VoronoiBrightnessImage,
    regularization=al.reg.AdaptiveBrightness,
)

"""
__MASS TOTAL PIPELINE (no lens light)__

The MASS TOTAL PIPELINE (no lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIPELINE to initialize the model priors. In this example it:

 - Uses an `EllPowerLaw` model for the lens's total mass distribution [The centre if unfixed from (0.0, 0.0)].
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS PIPELINE.
"""
analysis = al.AnalysisImaging(
    dataset=imaging, hyper_dataset_result=source_inversion_results.last
)

mass_results = slam.mass_total.no_lens_light(
    settings_autofit=settings_autofit,
    setup_hyper=setup_hyper,
    analysis=analysis,
    source_results=source_inversion_results,
    mass=af.Model(al.mp.EllPowerLaw),
)

slam.extensions.stochastic_fit(
    result=mass_results.last, analysis=analysis, **settings_autofit.fit_dict
)

"""
Finish.
"""
