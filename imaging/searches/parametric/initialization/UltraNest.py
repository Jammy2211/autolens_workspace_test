"""
Modeling: Mass Total + Source Parametric
========================================

This script gives a profile of a `UltraNest` model-fit to an `Imaging` dataset where the lens model is initialized,
where:

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
stepsampler_cls = None
stepsampler_cls = "RegionMHSampler"
# stepsampler_cls = "AHARMSampler"
# stepsampler_cls = "CubeMHSampler"
# stepsampler_cls = "CubeSliceSampler"
# stepsampler_cls = "RegionSliceSampler"

search = af.UltraNest(
    path_prefix=path_prefix,
    name=f"UltraNest_{stepsampler_cls}",
    unique_tag=dataset_name,
    stepsampler_cls=stepsampler_cls,
    nsteps=5,
    show_status=False,
    min_num_live_points=50,
    iterations_per_update=50000,
)

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
# imaging = imaging.apply_settings(settings=al.SettingsImaging(grid_class=al.Grid2DIterate))

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Model + Search + Analysis + Model-Fit__
"""

lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic)
source.bulge.intensity = af.UniformPrior(lower_limit=1e-2, upper_limit=1e2)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisImaging(dataset=imaging)

result = search.fit(model=model, analysis=analysis)

"""
Finished.
"""
