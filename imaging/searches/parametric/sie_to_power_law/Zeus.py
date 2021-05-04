"""
Chaining: SIE to Power-law
==========================

This script gives a profile of a `Zeus` model-fit to an `Imaging` dataset where the lens model is a power-law
that is initialized from an SIE, where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `EllPowerLaw`.
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

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Paths__
"""
# dataset_name = "mass_power_law__source_sersic"
dataset_name = "mass_power_law__source_sersic_compact"
path_prefix = path.join("searches", "parametric", "sie_to_power_law")

"""
__Search (Search Final)__
"""
search_2 = af.Zeus(
    path_prefix=path_prefix,
    name="Zeus",
    unique_tag=dataset_name,
    nwalkers=40,
    nsteps=2000,
    light_mode=False,
)


"""
__Dataset + Masking__ 
"""
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Model + Search + Analysis + Model-Fit (Search 1)__
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_1 = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]_sie", unique_tag=dataset_name, nlive=50
)

analysis = al.AnalysisImaging(dataset=imaging)

result_1 = search_1.fit(model=model, analysis=analysis)

"""
__Model + Analysis + Model-Fit (Search 2)__
"""
source = result_1.model.galaxies.source

mass = af.Model(al.mp.EllPowerLaw)
mass.take_attributes(result_1.model.galaxies.lens.mass)
mass.slope = af.UniformPrior(lower_limit=1.0, upper_limit=3.0)
shear = result_1.model.galaxies.lens.shear

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisImaging(dataset=imaging)

result_2 = search_2.fit(model=model, analysis=analysis)
