"""
Json: Simple
============

This script is a simple test of whether outputting a model to json and loading it does not lead to errors.

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `EllSersic`.
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

import json
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__Model (including json output and load)__
"""

lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

model_path = path.join("json")
model_file = path.join(model_path, "simple.json")

with open(model_file, "w+") as f:
    json.dump(model.dict, f, indent=4)

model = af.Collection.from_json(file=model_file)

"""
__Paths__
"""
dataset_name = "mass_power_law__source_sersic"
path_prefix = path.join("json")

"""
__Dataset + Masking__
"""
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.05,
)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Search + Analysis + Model-Fit__
"""
analysis = al.AnalysisImaging(dataset=imaging)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="simple",
    unique_tag=dataset_name,
)

result = search.fit(model=model, analysis=analysis)

"""
Finished.
"""
