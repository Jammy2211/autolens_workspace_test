"""
Database: Dataset x2
====================

This is an example of a model-fit performed to one dataset, but using custom settings on the second model-fit which 
augment and alter the data.. Each model-fit should be given a unique entry in the `.sqlite` database file.

(For the implementation below, the database uses the different output paths of the results to write two unique
entries. This would not be possible if we write straight to the database).

NOTES FOR RICH:

This is example of a model-fit which the tagging feature would have distinguished the two fits. The model and model 
priors of both fits are identical, so we can only distinguish them by inspecting the data.
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
__Model__
"""
lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllIsothermal, shear=al.mp.ExternalShear)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
DATASET:

__Dataset (No customization)__
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
__Masking + Search + Analysis + Model-Fit__
"""
search = af.DynestyStatic(
    name="search_dataset_normal",
    path_prefix=path.join("imaging", "database", "dataset_custom"),
    nlive=50,
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

search.fit(model=model, analysis=analysis)

"""
DATASET:

__Dataset (with customization)__
"""
masked_imaging = al.MaskedImaging(
    imaging=imaging,
    mask=mask,
    settings=al.SettingsImaging(
        signal_to_noise_limit=10.0, signal_to_noise_limit_radii=0.3
    ),
)

search = af.DynestyStatic(
    name="search_dataset_custom",
    path_prefix=path.join("imaging", "database", "dataset_custom"),
    nlive=50,
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

search.fit(model=model, analysis=analysis)

"""
__Database__

Add results to database.
"""
from autofit.database.aggregator import Aggregator

database_file = path.join(
    "output", "imaging", "database", "dataset_custom", "database.sqlite"
)

if path.isfile(database_file):
    os.remove(database_file)

agg = Aggregator.from_database(database_file)

agg.add_directory(path.join("output", "imaging", "database", "dataset_custom"))

agg = Aggregator.from_database(database_file)

"""
Check Aggregator works (This should load two mp_instances).
"""
agg_query = agg.query(agg.galaxies.lens.mass == al.mp.EllIsothermal)
mp_instances = [samps.median_pdf_instance for samps in agg.values("samples")]
print(mp_instances)
