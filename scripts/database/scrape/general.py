"""
Database: Model-Fit
===================

Tests that general results can be loaded from hard-disk via a database built via a scrape.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import os
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset + Masking__
"""
dataset_name = "no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

masked_dataset = dataset.apply_mask(mask=mask)

"""
__Model__
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search + Analysis + Model-Fit__
"""
name = "general"

search = af.DynestyStatic(
    name=name,
    path_prefix=path.join("database", "scrape"),
    unique_tag=dataset_name,
    nlive=50,
    maxcall=100,
    maxiter=100,
)

analysis = al.AnalysisImaging(dataset=masked_dataset)

result = search.fit(model=model, analysis=analysis, info={"hi": "there"})

"""
__Database__

Add results to database.
"""
from autofit.database.aggregator import Aggregator

database_file = "database_directory_general.sqlite"

try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass

agg = Aggregator.from_database(database_file)
agg.add_directory(path.join("output", "database", "scrape", dataset_name, name))

assert len(agg) > 0

"""
__Samples + Results__

Make sure database + agg can be used.
"""
for samples in agg.values("samples"):
    print(samples.log_likelihood_list[9])

ml_vector = [
    samps.max_log_likelihood(as_instance=False) for samps in agg.values("samples")
]
print(ml_vector, "\n\n")

"""
__Queries__
"""
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

for model in agg.values("model"):
    print(model.info)

for search in agg.values("search"):
    print(search)

for samples_summary in agg.values("samples_summary"):
    instance = samples_summary.max_log_likelihood()

for info in agg.values("info"):
    print(info["hi"])

for data in agg.values("dataset.data"):
    print(data)

for noise_map in agg.values("dataset.noise_map"):
    print(noise_map)

# for covariance in agg.values("covariance"):
#     print(covariance)


"""
__Aggregator Module__
"""
tracer_agg = al.agg.TracerAgg(aggregator=agg)
tracer_gen = tracer_agg.max_log_likelihood_gen_from()

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

for tracer in tracer_gen:

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(convergence=True, potential=True)

    print("Tracer Checked")


imaging_agg = al.agg.ImagingAgg(aggregator=agg)
imaging_gen = imaging_agg.dataset_gen_from()

for dataset in imaging_gen:

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    print("Imaging Checked")

fit_agg = al.agg.FitImagingAgg(
    aggregator=agg,
    settings_dataset=al.SettingsImaging(sub_size=4),
    settings_pixelization=al.SettingsPixelization(use_border=False),
)
fit_imaging_gen = fit_agg.max_log_likelihood_gen_from()

for fit in fit_imaging_gen:

    fit_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_plotter.subplot_fit()

    print("FitImaging Checked")


