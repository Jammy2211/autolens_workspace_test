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
 - The lens galaxy's dark matter mass distribution is modeled as a `NFWMCRLudlow`.
 - The source galaxy's light is a parametric `Inversion`.

This runner uses the SLaM pipelines:

 `source_lp`
 `source_pix/with_lens_light`
 `light_lp`
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
        "light_sersic__mass_light_dark__source_pix",
        "sensitivity",
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

bulge = af.Model(al.lp.Sersic)
disk = af.Model(al.lp.Sersic)
bulge.centre = disk.centre

source_lp_result = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=bulge,
    lens_disk=disk,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.Sersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__SOURCE PIX PIPELINE (with lens light)__

The SOURCE PIX PIPELINE (with lens light) uses two searches to initialize a robust model for the pixelization
that reconstructs the source galaxy's light. It begins by fitting a `Voronoi` pixelization with `Constant` 
regularization, to set up the model and hyper images, and then:

 - Uses a `Voronoi` pixelization.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
 SOURCE PIX PIPELINE.
"""

analysis = al.AnalysisImaging(
    dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_lp_result)
)

source_pix_results = slam.source_pix.run(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    image_mesh=al.image_mesh.Hilbert,
    mesh=al.mesh.Voronoi,
    regularization=al.reg.AdaptiveBrightnessSplit,
)


"""
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE PIX PIPELINE.
In this example it:

 - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
 light [Do not use the results of the SOURCE LP PIPELINE to initialize priors].

 - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE LP PIPELINE].

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values].
"""
bulge = af.Model(al.lp.Sersic)
disk = af.Model(al.lp.Sersic)
bulge.centre = disk.centre

light_results = slam.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result=source_pix_results,
    lens_bulge=bulge,
    lens_disk=disk,
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

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the MASS 
 LIGHT DARK PIPELINE.
"""
analysis = al.AnalysisImaging(
    dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1)
)

lens_bulge = af.Model(al.lmp.Sersic)
dark = af.Model(al.mp.NFWMCRLudlow)

dark.centre = lens_bulge.centre

mass_results = slam.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    source_results=source_pix_results,
    light_results=light_results,
    lens_bulge=lens_bulge,
    lens_disk=af.Model(al.lmp.Sersic),
    dark=dark,
)

"""
__SUBHALO PIPELINE (sensitivity mapping)__

The SUBHALO PIPELINE (sensitivity mapping) performs sensitivity mapping of the data using the lens model
fitted above, so as to determine where subhalos of what mass could be detected in the data. A full description of
Sensitivity mapping if given in the SLaM pipeline script `slam/subhalo/sensitivity_imaging.py`.

Each model-fit performed by sensitivity mapping creates a new instance of an `Analysis` class, which contains the
data simulated by the `simulate_cls` for that model. This requires us to write a wrapper around the 
PyAutoLens `AnalysisImaging` class.
"""
subhalo_results = slam.subhalo.sensitivity_imaging_pix.run(
    settings_search=settings_search,
    mask=mask,
    psf=dataset.psf,
    mass_results=mass_results,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
)

"""
Finish.
"""
