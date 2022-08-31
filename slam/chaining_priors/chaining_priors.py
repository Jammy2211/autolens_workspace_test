"""
SLaM (Source, Light and Mass): Light Parametric + Mass Total + Source Parametric
================================================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `EllSersic` profile for the bulge, this will be used in the subsequent MASS PIPELINE.

Using a SOURCE PARAMETRIC PIPELINE, LIGHT PIPELINE and a MASS PIPELINE this SLaM script fits `Imaging` of a strong
lens system, where in the final model:

 - The lens galaxy's light is a bulge+disk `EllSersic` and `EllExponential`.
 - The lens galaxy's total mass distribution is an `EllPowerLaw`.
 - The source galaxy's light is a parametric `EllSersic`.

This runner uses the SLaM pipelines:

 `source_parametric/source_parametric__with_lens_light`
 `light_parametric/with_lens_light`
 `mass_total/mass_total__with_lens_light`

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

conf.instance.push(new_path=path.join(cwd, "config", "searches"))

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

settings_autofit = af.SettingsSearch(
    path_prefix=path.join(
        "slam", "light_sersic__mass_total__source_parametric", "hyper_all"
    ),
    unique_tag=dataset_name,
    number_of_cores=1,
    session=None,
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
    search_pixelization_dict={"nlive": 30, "sample": "rwalk"},
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

lens_bulge = af.Model(al.lp.EllSersicCore)
lens_bulge.radius_break = 0.05
lens_bulge.gamma = 0.0
lens_bulge.alpha = 2.0

lens_disk = af.Model(al.lp.EllExponentialCore)
lens_disk.radius_break = 0.05
lens_disk.gamma = 0.0
lens_disk.alpha = 2.0

lens_bulge.centre = lens_disk.centre

source_bulge = af.Model(al.lp.EllSersicCore)
source_bulge.radius_break = 0.05
source_bulge.gamma = 0.0
source_bulge.alpha = 2.0


source_parametric_results = slam.source_parametric.with_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    lens_bulge=lens_bulge,
    lens_disk=lens_disk,
    mass=mass,
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=0.5,
    redshift_source=1.0,
)


"""
Finish.
"""
