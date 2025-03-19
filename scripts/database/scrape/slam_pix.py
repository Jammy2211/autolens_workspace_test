"""
Database: Model-Fit
===================

This is a simple example of a model-fit which we wish to write to the database. This should simply output the
results to the `.sqlite` database file.
"""
import pytest

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from astropy.io import fits
import os
from os import path

os.environ["PYAUTOFIT_TEST_MODE"] = "1"

cwd = os.getcwd()
from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "fit"))

import autofit as af
import autolens as al
import autolens.plot as aplt
import slam

"""
__Dataset + Masking__
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

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=path.join("database", "scrape", "slam_pix"),
    number_of_cores=1,
    session=None,
    info={"hi": "there"},
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
 
 - Uses a parametric `Sersic` bulge and `Exponential` disk with centres aligned for the lens
 galaxy's light.
 
 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

 Settings:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=dataset)

bulge = af.Model(al.lp_linear.Sersic)
disk = af.Model(al.lp_linear.Exponential)
# disk = af.Model(al.lp_linear.Sersic)
bulge.centre = disk.centre

source_lp_result = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=bulge,
    lens_disk=disk,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp_linear.Sersic),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)


"""
__SOURCE PIX PIPELINE (with lens light)__

The SOURCE PIX PIPELINE (with lens light) uses two searches to initialize a robust model for the pixelization
that reconstructs the source galaxy's light. It begins by fitting a `Delaunay` pixelization with `Constant` 
regularization, to set up the model and hyper images, and then:

 - Uses a `Delaunay` pixelization.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
 SOURCE PIX PIPELINE.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
)

source_pix_result_1 = slam.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=al.mesh.Delaunay,
)

"""
__SOURCE PIX PIPELINE 2 (with lens light)__
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    settings_inversion=al.SettingsInversion(
        image_mesh_min_mesh_pixels_per_pixel=3,
        image_mesh_min_mesh_number=5,
        image_mesh_adapt_background_percent_threshold=0.1,
        image_mesh_adapt_background_percent_check=0.8,
    ),
)

source_pix_result_2 = slam.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    image_mesh=al.image_mesh.Hilbert,
    mesh=al.mesh.Delaunay,
    regularization=al.reg.AdaptiveBrightnessSplit,
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
bulge = af.Model(al.lp_linear.Sersic)
disk = af.Model(al.lp_linear.Exponential)
bulge.centre = disk.centre

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
)

light_result = slam.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
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
    dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1)
)

multipole = af.Model(al.mp.PowerLawMultipole)
multipole.m = 3

mass_result = slam.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
    multipole_3=multipole,
)


"""
__Database__

Add results to database.
"""
from autofit.database.aggregator import Aggregator

database_file = "database_directory_slam_pix.sqlite"

try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass

agg = Aggregator.from_database(database_file)
agg.add_directory(directory=path.join("output", "database", "scrape", "slam_pix"))

assert len(agg) > 0

"""
__Samples + Results__

Make sure database + agg can be used.
"""
print("\n\n***********************")
print("****RESULTS TESTING****")
print("***********************\n")

for samples in agg.values("samples"):
    print(samples.parameter_lists[0])

mp_instances = [samps.median_pdf() for samps in agg.values("samples")]
print(mp_instances)

"""
__Queries__
"""
print("\n\n***********************")
print("****QUERIES TESTING****")
print("***********************\n")

unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "mass_sie__source_sersic__1")
samples_gen = agg_query.values("samples")

unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "incorrect_name")
samples_gen = agg_query.values("samples")

name = agg.search.name
agg_query = agg.query(name == "database_example")
print("Total Queried Results via search name = ", len(agg_query), "\n\n")

lens = agg.model.galaxies.lens
agg_query = agg.query(lens.mass == al.mp.Isothermal)
print("Total Samples Objects via `Isothermal` model query = ", len(agg_query), "\n")

source = agg.model.galaxies.source
agg_query = agg.query(source.disk == None)
print("Total Samples Objects via `Isothermal` model query = ", len(agg_query), "\n")

mass = agg.model.galaxies.lens.mass
agg_query = agg.query((mass == al.mp.Isothermal) & (mass.einstein_radius > 1.0))

print(
    "Total Samples Objects In Query `Isothermal and einstein_radius > 3.0` = ",
    len(agg_query),
    "\n",
)

"""
__Files__

Check that all other files stored in database (e.g. model, search) can be loaded and used.
"""
print("\n\n***********************")
print("*****FILES TESTING*****")
print("***********************\n")

for model in agg.values("model"):
    print(f"\n****Model Info (model)****\n\n{model.info}")
    assert model.info[0] == "T"

for search in agg.values("search"):
    print(f"\n****Search (search)****\n\n{search}")
    assert "[" in search.paths.name

for samples_summary in agg.values("samples_summary"):
    instance = samples_summary.max_log_likelihood()
    print(f"\n****Max Log Likelihood (samples_summary)****\n\n{instance}")
    assert instance.galaxies.lens.mass.einstein_radius > 0.0

for info in agg.values("info"):
    print(f"\n****Info****\n\n{info}")
    assert info["hi"] == "there"

for data in agg.values("dataset.data"):
    print(f"\n****Data (dataset.data)****\n\n{data}")
    assert isinstance(data[0], fits.PrimaryHDU)

for noise_map in agg.values("dataset.noise_map"):
    print(f"\n****Noise Map (dataset.noise_map)****\n\n{noise_map}")
    assert isinstance(noise_map[0], fits.PrimaryHDU)

for covariance in agg.values("covariance"):
    print(f"\n****Covariance (covariance)****\n\n{covariance}")
    assert covariance is not None


"""
__Aggregator Module__
"""
print("\n\n***********************")
print("***AGG MODULE TESTING***")
print("***********************\n\n")

agg = agg.query(agg.search.name == "mass_total[1]")

tracer_agg = al.agg.TracerAgg(aggregator=agg)
tracer_gen = tracer_agg.max_log_likelihood_gen_from()

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

for tracer_list in tracer_gen:
    # Only one `Analysis` so take first and only tracer.
    tracer = tracer_list[0]

    try:
        tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
        tracer_plotter.figures_2d(convergence=True, potential=True)

    except al.exc.ProfileException:
        print("TracerAgg with linear light profiles raises correct ProfileException")

    assert tracer.galaxies[0].mass.einstein_radius > 0.0

    print("TracerAgg Checked")

imaging_agg = al.agg.ImagingAgg(aggregator=agg)
imaging_gen = imaging_agg.dataset_gen_from()

for dataset_list in imaging_gen:
    dataset = dataset_list[0]

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    assert dataset.pixel_scales[0] > 0.0

    print("ImagingAgg Checked")

fit_agg = al.agg.FitImagingAgg(
    aggregator=agg,
    settings_inversion=al.SettingsInversion(use_border_relocator=False),
)
fit_imaging_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_imaging_gen:
    fit = fit_list[0]

    fit_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_plotter.subplot_fit()

    assert fit.tracer.galaxies[0].mass.einstein_radius > 0.0

    print("FitImagingAgg Checked")

fit_imaging_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_imaging_gen:
    fit = fit_list[0]

    assert fit.adapt_images.model_image is not None

    print("FitImagingAgg Adapt Images Checked")


"""
__CSV__

Check that the `AggregateCSV` object can be used to output results to a `.csv` file.
"""
agg_csv = af.AggregateCSV(aggregator=agg)

agg_csv.add_column(argument="galaxies.galaxy.bulge.effective_radius")
agg_csv.add_column(argument="galaxies.galaxy.bulge.sersic_index")


os.environ["PYAUTOFIT_TEST_MODE"] = "0"
