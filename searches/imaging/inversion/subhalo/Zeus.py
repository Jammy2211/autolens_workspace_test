"""
Chaining: Subhalo
=================

This script gives a profile of a `DynestyStatic` model-fit to an `Imaging` dataset where the lens model is a power-law
and subhalo, where

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `PowerLaw` and `ExternalShear`.
 - The subhalo is at the lens redshfit and is an `NFWMCRLudlow`.
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
dataset_name = "mass_sie__subhalo_nfw__source_sersic"
path_prefix = path.join("searches", "inversion", "subhalo", dataset_name)

"""
__Search (Search Final)__
"""
search_4 = af.Zeus(
    path_prefix=path_prefix,
    name="Zeus",
    unique_tag=dataset_name,
    nwalkers=30,
    nsteps=400,
    iterations_per_update=20,
)


"""
__Dataset + Masking__ 
"""
dataset_path = path.join("dataset", "imaging", "subhalo", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.5
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model + Search + Analysis + Model-Fit (Search 1)__
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_1 = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]_sie", unique_tag=dataset_name, nlive=50
)

analysis = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__
"""
source = result_1.model.galaxies.source

mass = af.Model(al.mp.PowerLaw)
mass.take_attributes(result_1.model.galaxies.lens.mass)
shear = result_1.model.galaxies.lens.shear

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_2 = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[2]_power_law",
    unique_tag=dataset_name,
    nlive=50,
)

analysis = al.AnalysisImaging(dataset=dataset)

result_2 = search_2.fit(model=model, analysis=analysis)


"""
__Model + Analysis + Model-Fit (Search 3)__
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=result_2.instance.galaxies.lens.mass,
    shear=result_2.instance.galaxies.lens.shear,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=al.mesh.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant,
)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_3 = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[3]_inversion",
    unique_tag=dataset_name,
    nlive=50,
)

analysis = al.AnalysisImaging(dataset=dataset)

result_3 = search_3.fit(model=model, analysis=analysis)


"""
__Model + Analysis + Model-Fit (Search 4)__
"""

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=result_2.model.galaxies.lens.mass,
    shear=result_2.model.galaxies.lens.shear,
)

subhalo = af.Model(
    al.Galaxy,
    redshift=result_1.instance.galaxies.lens.redshift,
    mass=al.mp.NFWMCRLudlow,
)

subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
subhalo.mass.centre_0 = af.UniformPrior(lower_limit=0.8, upper_limit=2.4)
subhalo.mass.centre_1 = af.UniformPrior(lower_limit=-0.8, upper_limit=0.8)

subhalo.mass.redshift_object = result_1.instance.galaxies.lens.redshift
subhalo.mass.redshift_source = result_1.instance.galaxies.source.redshift

model = af.Collection(
    galaxies=af.Collection(
        lens=lens, source=result_3.instance.galaxies.source, subhalo=subhalo
    )
)

analysis = al.AnalysisImaging(dataset=dataset)

result_4 = search_4.fit(model=model, analysis=analysis)
