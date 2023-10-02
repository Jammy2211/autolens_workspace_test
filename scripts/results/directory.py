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
search = af.Nautilus(
    name="mass[sie]_source[bulge]",
    path_prefix=path.join("build", "results", "directory"),
    unique_tag=dataset_name,
    n_live=50,
)

analysis = al.AnalysisImaging(dataset=masked_dataset)

search.fit(model=model, analysis=analysis)

"""
__Database__

Add results to database.
"""
from autofit.database.aggregator import Aggregator

database_file = "result_directory.sqlite"

try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass

agg = Aggregator.from_database(database_file)
agg.add_directory(path.join("output", "build", "results", "directory"))

"""
Check Aggregator works (This should load one mp_instance).
"""
for model in agg.values("model"):
    print(model)

for search in agg.values("search"):
    print(search)

for samples in agg.values("samples"):
    print(samples)

for samples_info in agg.values("samples_info"):
    print(samples_info)

for tracer in agg.values("tracer"):
    print(tracer)

for data in agg.values("dataset.data"):
    print(data)

for noise_map in agg.values("dataset.noise_map"):
    print(noise_map)

for psf in agg.values("dataset.psf"):
    print(psf)

for settings in agg.values("dataset.settings"):
    print(settings)

for mask in agg.values("dataset.mask"):
    print(mask)

for settings_pixelization in agg.values("settings_pixelization"):
    print(settings_pixelization)

for settings_inversion in agg.values("settings_inversion"):
    print(settings_inversion)

for cosmology in agg.values("cosmology"):
    print(cosmology)

for covariance in agg.values("covariance"):
    print(covariance)
    
tracer_agg = al.agg.TracerAgg(aggregator=agg)
tracer_gen = tracer_agg.max_log_likelihood_gen_from()

for tracer in tracer_gen:
    
    print(tracer)

imaging_agg = al.agg.ImagingAgg(aggregator=agg)
imaging_gen = imaging_agg.dataset_gen_from()

for dataset in imaging_gen:
    
    print(dataset)

fit_agg = al.agg.FitImagingAgg(
    aggregator=agg,
    settings_dataset=al.SettingsImaging(sub_size=4),
    settings_pixelization=al.SettingsPixelization(use_border=False),
)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit in fit_gen:

    print(fit)
