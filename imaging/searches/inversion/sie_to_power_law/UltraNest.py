"""
Chaining: SIE to Power-law
==========================

This script gives a profile of a `DynestyStatic` model-fit to an `Imaging` dataset where the lens model is a power-law
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
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
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
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=result_1.instance.galaxies.lens.mass,
    shear=result_1.instance.galaxies.lens.shear,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant,
)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_2 = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[2]_inversion",
    unique_tag=dataset_name,
    nlive=50,
)

analysis = al.AnalysisImaging(dataset=imaging)

result_2 = search_2.fit(model=model, analysis=analysis)

"""
__Model + Analysis + Model-Fit (Search 3)__
"""
mass = af.Model(al.mp.EllPowerLaw)
mass.take_attributes(result_1.model.galaxies.lens.mass)

lens = af.Model(
    al.Galaxy, redshift=0.5, mass=mass, shear=result_1.model.galaxies.lens.shear
)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=result_2.instance.galaxies.source)
)

settings_lens = al.SettingsLens(
    positions_threshold=result_2.positions_threshold_from(
        factor=3.0, minimum_threshold=0.2
    )
)

analysis = al.AnalysisImaging(
    dataset=imaging,
    positions=result_2.image_plane_multiple_image_positions,
    settings_lens=settings_lens,
)

result_3 = search_3.fit(model=model, analysis=analysis)
