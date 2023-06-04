"""
Modeling: Mass Total + Source Parametric
========================================

This script gives a profile of a `DynestyStatic` model-fit to an `Imaging` dataset where the lens model is initialized,
where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `Sersic`.
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

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Paths__
"""
dataset_name = "mass_power_law__source_sersic"
# dataset_name = "mass_power_law__source_sersic_compact"
path_prefix = path.join("searches", "parametric", "initialization")

"""
__Search__
"""
search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="DynestyStatic",
    unique_tag=dataset_name,
    nlive=50,
    walks=10,
    iterations_per_update=5000,
)

"""
__Dataset + Masking__
"""
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf_big.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.05,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model + Search + Analysis + Model-Fit__
"""

lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
Finished.
"""
