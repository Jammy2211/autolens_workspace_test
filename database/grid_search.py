"""
Database: Model-Fit
===================

This is a simple example of a model-fit which we wish to write to the database. This should simply output the
results to the `.sqlite` database file.
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

"""
__Dataset + Masking__
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

masked_imaging = imaging.apply_mask(mask=mask)

"""
__Model__
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic)

subhalo = af.Model(al.Galaxy, redshift=0.5, mass=af.Model(al.mp.SphNFWMCRLudlow))

subhalo.mass.centre_0 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
subhalo.mass.centre_1 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

subhalo.mass.redshift_object = 0.5
subhalo.mass.redshift_source = 1.0

model = af.Collection(galaxies=af.Collection(lens=lens, source=source, subhalo=subhalo))

"""
__Search + Analysis + Model-Fit__
"""
analysis = al.AnalysisImaging(dataset=masked_imaging)

search = af.DynestyStatic(
    path_prefix=path.join("database", "grid_search"),
    name="mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    nlive=50,
)

subhalo_grid_search = al.SubhaloSearch(
    grid_search=af.SearchGridSearch(
        search=search, number_of_steps=2, number_of_cores=1
    ),
    result_no_subhalo=None,
)

subhalo_search_result = subhalo_grid_search.fit(
    model=model,
    analysis=analysis,
    grid_priors=[
        model.galaxies.subhalo.mass.centre_0,
        model.galaxies.subhalo.mass.centre_1,
    ],
)

"""
__Database__

Add results to database.
"""
from autofit.database.aggregator import Aggregator

database_file = path.join("output", "database", "grid_search", "database.sqlite")

if path.isfile(database_file):
    os.remove(database_file)

agg = Aggregator.from_database(database_file)

agg.add_directory(path.join("output", "database", "grid_search"))

# agg = Aggregator.from_database(database_file)

"""
Check Aggregator works (This should load one mp_instance).
"""
print(len(agg))

print(agg.subhalo_search_result)

agg_grid = agg.grid_searches()
print("Total `agg_grid` = ", len(agg_grid), "\n")
