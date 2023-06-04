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
dataset_name = "light_sersic__mass_sie__source_sersic_compact"
dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)
path_prefix = path.join("searches", "grid_iterate")

"""
__Search__
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="cored", unique_tag=dataset_name, nlive=100, walks=10
)

"""
__Dataset + Masking__
"""

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
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

dataset = dataset.apply_settings(
    settings=al.SettingsImaging(grid_class=al.Grid2D, sub_size=1)
)

"""
__Model + Search + Analysis + Model-Fit__
"""

bulge = af.Model(al.lp.SersicCore)
bulge.radius_break = 0.001
bulge.gamma = 0.0
bulge.alpha = 2.0

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
Finished.
"""
