"""
SLaM (Source, Light and Mass): Light Parametric + Mass Total + Source Parametric
================================================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS PIPELINE.

Using a SOURCE LP PIPELINE, LIGHT PIPELINE and a MASS PIPELINE this SLaM script fits `Imaging` of a strong
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

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.2,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

masked_dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(
    imaging=masked_dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("database", "slam_session_simple")

"""
___Session__

To output results directly to the database, we start a session, which includes the name of the database `.sqlite` file
where results are stored.
"""
session = af.db.open_database("database.sqlite")

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, databse use, etc.
"""
settings_autofit = af.SettingsSearch(
    path_prefix=path.join("database", "session", "slam_simple"),
    unique_tag=dataset_name,
    number_of_cores=1,
    session=session,
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
of different models in the LIGHT PIPELINE and MASS PIPELINE can be performed consistently.
"""
setup_adapt = al.SetupAdapt(
    hyper_galaxies_lens=True,
    hyper_galaxies_source=True,
    hyper_image_sky=al.hyper_data.HyperImageSky,
    hyper_background_noise=al.hyper_data.HyperBackgroundNoise,
)

"""
__SOURCE LP PIPELINE (with lens light)__

The SOURCE LP PIPELINE (with lens light) uses three searches to initialize a robust model for the 
source galaxy's light, which in this example:

 - Uses a parametric `Sersic` bulge and `Exponential` disk with centres aligned for the lens
 galaxy's light.

 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

 Settings:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=masked_dataset)

bulge = af.Model(al.lp.Sersic)
disk = af.Model(al.lp.Exponential)
bulge.centre = disk.centre

source_lp_results = slam.source_lp.run(
    settings_autofit=settings_autofit,
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
analysis = al.AnalysisImaging(
    dataset=masked_dataset, adapt_result=source_lp_results.last
)

bulge = af.Model(al.lp.Sersic)
disk = af.Model(al.lp.Exponential)
bulge.centre = disk.centre

light_results = slam.light_lp.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_results=source_lp_results,
    lens_bulge=bulge,
    lens_disk=disk,
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
    dataset=masked_dataset, adapt_result=source_lp_results.last
)

mass_results = slam.mass_total.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_results=source_lp_results,
    light_results=light_results,
    mass=af.Model(al.mp.PowerLaw),
)

"""
Tests that queries work.
"""
agg = af.Aggregator.from_database("database.sqlite", completed_only=True)

name = agg.search.name
agg_query = agg.query(name == "mass_total[1]_mass[total]_source")
samples_gen = agg_query.values("samples")
print(
    "Total Samples Objects via `Isothermal` model query = ",
    len(list(samples_gen)),
    "\n",
)


lens = agg.model.galaxies.lens
agg_query = agg.query(lens.mass == al.mp.Isothermal)
samples_gen = agg_query.values("samples")
print(
    "Total Samples Objects via `Isothermal` model query = ",
    len(list(samples_gen)),
    "\n",
)


fit_imaging_agg = al.agg.FitImagingAgg(aggregator=agg)
fit_imaging_gen = fit_imaging_agg.max_log_likelihood_gen_from()
print(list(fit_imaging_gen)[0].figure_of_merit)

"""
Finish.
"""
