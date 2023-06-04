"""
Chaining: SIE to Power-law
==========================

This script gives a profile of a `DynestyStatic` model-fit to an `Imaging` dataset where the lens model is a power-law
that is initialized from an SIE, where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `PowerLaw`.
 - The source galaxy's light is a parametric `Sersic`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path
import sys

cwd = os.getcwd()
sys.path.insert(0, os.getcwd())
from slam import extensions

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "searches"))

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Paths__
"""
dataset_name = "simple_lens"

path_prefix = path.join("searches", "inversion", "lens_light_sersic_x2")

"""
__Search (Search Final)__
"""
search_5 = af.DynestyStatic(
    path_prefix=path_prefix, name="DynestyStatic", unique_tag=dataset_name, nlive=100
)


"""
__Dataset + Masking__ 
"""
dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Adapt Setup__

This tests uses the galaxies feature on the lens.
"""
setup_adapt = al.SetupAdapt(
    hyper_galaxies_lens=True,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
)

"""
__Model + Search + Analysis + Model-Fit (Search 1)__
"""
bulge = af.Model(al.lp.Sersic)
disk = af.Model(al.lp.Exponential)
bulge.centre = disk.centre

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, disk=disk)
model = af.Collection(galaxies=af.Collection(lens=lens))

search_1 = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[1]_lens_light",
    unique_tag=dataset_name,
    nlive=50,
)

analysis = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 1)__
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=result_1.instance.galaxies.lens.bulge,
    disk=result_1.instance.galaxies.lens.disk,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_2 = af.DynestyStatic(
    path_prefix=path_prefix, name="search[2]_sie", unique_tag=dataset_name, nlive=75
)

analysis = al.AnalysisImaging(dataset=dataset)

result_2 = search_2.fit(model=model, analysis=analysis)

"""
__Model + Analysis + Model-Fit (Search 3)__
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=result_1.model.galaxies.lens.bulge,
    disk=result_1.model.galaxies.lens.disk,
    mass=result_2.model.galaxies.lens.mass,
    shear=result_2.model.galaxies.lens.shear,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=result_2.model.galaxies.source.bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_3 = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[3]_parametric_both",
    unique_tag=dataset_name,
    nlive=75,
)

analysis = al.AnalysisImaging(dataset=dataset)

result_3 = search_3.fit(model=model, analysis=analysis)

result_3 = extensions.adapt_fit(
    setup_adapt=setup_adapt,
    result=result_3,
    analysis=analysis,
    include_hyper_image_sky=False,
)

"""
__Model + Analysis + Model-Fit (Search 4)__
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=result_3.instance.galaxies.lens.bulge,
    disk=result_3.instance.galaxies.lens.disk,
    mass=result_3.instance.galaxies.lens.mass,
    shear=result_3.instance.galaxies.lens.shear,
    hyper_galaxy=setup_adapt.hyper_galaxy_lens_from(result=result_3),
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=al.mesh.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant,
)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_4 = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[4]_inversion",
    unique_tag=dataset_name,
    nlive=30,
)

analysis = al.AnalysisImaging(dataset=dataset, adapt_result=result_3)

result_4 = search_4.fit(model=model, analysis=analysis)

result_4 = extensions.adapt_fit(
    setup_adapt=setup_adapt,
    result=result_4,
    analysis=analysis,
    include_hyper_image_sky=False,
)

"""
__Model + Analysis + Model-Fit (Search 5)__
"""
hyper_galaxy = setup_adapt.hyper_galaxy_lens_from(
    result=result_4, noise_factor_is_model=True
)

bulge = af.Model(al.lp.Sersic)
bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

disk = af.Model(al.lp.Sersic)
disk.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
disk.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    disk=disk,
    mass=result_3.instance.galaxies.lens.mass,
    shear=result_3.instance.galaxies.lens.shear,
    hyper_galaxy=hyper_galaxy,
)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=result_4.instance.galaxies.source)
)

analysis = al.AnalysisImaging(dataset=dataset, adapt_result=result_4)

result_5 = search_5.fit(model=model, analysis=analysis)
