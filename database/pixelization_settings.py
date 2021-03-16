"""
Database: Model-Fit
===================

This is an example of two model-fit using the same data and model, which fit a pixelization, which we wish to write to 
the database. 

One model-fit uses the pixelization setting `use_border=True`, whereas the other does not.

(For the implementation below, the database uses the different output paths of the results to write two unique
entries. This would not be possible if we write straight to the database).
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
__Model__
"""
lens = al.GalaxyModel(
    redshift=0.5, mass=al.mp.EllipticalIsothermal, shear=al.mp.ExternalShear
)
source = al.GalaxyModel(redshift=1.0, pixelization=al.pix.Rectangular, regularization=al.reg.Constant)

model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(lens=lens, source=source)
)

"""
__Search + Analysis + Model-Fit (Use Border)__
"""
search = af.DynestyStatic(
    name="pixelization_use_border",
    path_prefix=path.join("imaging", "database", "pixelization_settings"),
    n_live_points=50,
)

analysis = al.AnalysisImaging(
    dataset=masked_imaging,
    settings_pixelization=al.SettingsPixelization(use_border=True)
)

search.fit(model=model, analysis=analysis)

"""
__Search + Analysis + Model-Fit (Not Use Border)__
"""
search = af.DynestyStatic(
    name="pixelization_not_use_border",
    path_prefix=path.join("imaging", "database", "pixelization_settings"),
    n_live_points=50,
)

analysis = al.AnalysisImaging(
    dataset=masked_imaging,
    settings_pixelization=al.SettingsPixelization(use_border=False)
)

search.fit(model=model, analysis=analysis)

"""
__Database__

Add results to database.
"""
from autofit.database.aggregator import Aggregator

database_file = path.join("output", "imaging", "database", "pixelization_settings", "database.sqlite")

if path.isfile(database_file):
    os.remove(database_file)

agg = Aggregator.from_database(database_file)

agg.add_directory(path.join("output",  "imaging", "database", "pixelization_settings"))

agg = Aggregator.from_database(database_file)

"""
Check Aggregator works (This should load one mp_instance).
"""
agg_query = agg.query(agg.galaxies.lens.mass == al.mp.EllipticalIsothermal)
mp_instances = [samps.median_pdf_instance for samps in agg.values("samples")]
print(mp_instances)