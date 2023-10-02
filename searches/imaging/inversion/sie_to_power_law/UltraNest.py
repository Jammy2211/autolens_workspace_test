"""
Chaining: SIE to Power-law
==========================

This script gives a profile of a `Nautilus` model-fit to an `Imaging` dataset where the lens model is a power-law
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

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "searches"))

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Paths__
"""
# dataset_name = "with_lens_light_search"
dataset_name = "mass_power_law__source_sersic_compact"

path_prefix = path.join("searches", "inversion", "sie_to_power_law")

"""
__Search (Search Final)__
"""
stepsampler_cls = None
# stepsampler_cls = "RegionMHSampler"
# stepsampler_cls = "AHARMSampler"
# stepsampler_cls = "CubeMHSampler"
# stepsampler_cls = "CubeSliceSampler"
# stepsampler_cls = "RegionSliceSampler"

search_3 = af.UltraNest(
    path_prefix=path_prefix,
    #    name=f"UltraNest_{stepsampler_cls}",
    name=f"UltraNest",
    unique_tag=dataset_name,
    stepsampler_cls=stepsampler_cls,
    nsteps=5,
    min_num_live_points=50,
    iterations_per_update=10000,
)

"""
__Dataset + Masking__ 
"""
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
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

search_1 = af.Nautilus(
    path_prefix=path_prefix, name="search[1]_sie", unique_tag=dataset_name, n_live=50
)

analysis = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model, analysis=analysis)

"""
__Model + Analysis + Model-Fit (Search 2)__
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=result_1.instance.galaxies.lens.mass,
    shear=result_1.instance.galaxies.lens.shear,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=al.mesh.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant,
)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]_inversion",
    unique_tag=dataset_name,
    n_live=50,
)

analysis = al.AnalysisImaging(dataset=dataset)

result_2 = search_2.fit(model=model, analysis=analysis)

"""
__Model + Analysis + Model-Fit (Search 3)__
"""
mass = af.Model(al.mp.PowerLaw)
mass.take_attributes(result_1.model.galaxies.lens.mass)

lens = af.Model(
    al.Galaxy, redshift=0.5, mass=mass, shear=result_1.model.galaxies.lens.shear
)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=result_2.instance.galaxies.source)
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood=result_2.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

result_3 = search_3.fit(model=model, analysis=analysis)
