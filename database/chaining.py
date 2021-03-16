"""
Database: Chaining
==================

This is an example of a chained model-fit which uses two non-linear searches, both of which we wish to write to the
database.

The two searches with the exact same lens model, but using different priors (as the second search is passed priors
from the first search). The database must distinguish between these two fits based on the priors. Irrespective of
how many chains a model-fit uses.

(For the implementation below, the database uses the different output paths of the results to write two unique
entries. This would not be possible if we write straight to the database).

NOTES FOR RICH:

For complex pipelines, it is not possible to rely on the search `name` to determine if a chained search is unique.
There are pipelines which can take a different series of searches to reach some common search in the chain. In this case
we have two unique model-fits, both using the same search with the same name! This would have previously been
distinguished via the tagging strings.

The priors + instance variables of a model after a series of chained searches will always be unique (such that we can
always compare priors to  determine if a search is resuming from a previous run, or is a new unique model-fit that has
not been performed previously).
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

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

"""
__Analysis__
"""
analysis = al.AnalysisImaging(dataset=masked_imaging)

"""
__Model (Search 1)__
"""
lens = al.GalaxyModel(
    redshift=0.5, mass=al.mp.EllipticalIsothermal, shear=al.mp.ExternalShear
)
source = al.GalaxyModel(redshift=1.0, bulge=al.lp.EllipticalSersic)

model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(lens=lens, source=source)
)

"""
__Search + Model Fit (Search 1)__
"""
search = af.DynestyStatic(
    name="search[1]",
    path_prefix=path.join("imaging", "database", "chaining", dataset_name),
)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model (Search 2)__
"""
model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(lens=result_1.model.galaxies.lens, source=result_1.model.galaxies.source)
)

"""
__Search + Model Fit (Search 2)__
"""
search = af.DynestyStatic(
    name="search[2]",
    path_prefix=path.join("imaging", "database", "chaining", dataset_name),
)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Database__

Add results to database.
"""
from autofit.database.aggregator import Aggregator

database_file = path.join("output", "imaging", "database", "database.sqlite")

if path.isfile(database_file):
    os.remove(database_file)

agg = Aggregator.from_database(database_file)

agg.add_directory(path.join("output",  "imaging", "database", "chaining"))

agg = Aggregator.from_database(database_file)

"""
Check Aggregator works (This should load two mp_instances).
"""
agg_query = agg.query(agg.galaxies.lens.mass == al.mp.EllipticalIsothermal)
mp_instances = [samps.median_pdf_instance for samps in agg.values("samples")]
print(mp_instances)