"""
Viusalize: Imaging
==================

This script performs an imaging model fit, where all images are output during visualization as .png and .fits
files.

This tests all visualization outputs in **PyAutoLens** for imaging data.
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

conf.instance.push(new_path=path.join(cwd, "config", "visualizer"))

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and plot the strong lens dataset `mass_sie__source_sersic` via .fits files, which we will fit with the lens model.
"""
dataset_label = "build"
dataset_type = "imaging"
dataset_name = "no_lens_light"

dataset_path = path.join("dataset", dataset_label, dataset_type, dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Masking__
"""
mask_2d = al.Mask2D.circular_annular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, inner_radius=0.8, outer_radius=2.6
)

imaging = imaging.apply_mask(mask=mask_2d)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Positions__
"""
positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)

"""
__Model__
"""
bulge = af.Model(al.lp.SphDevVaucouleurs)

bulge.centre = (0.0, 0.0)
bulge.intensity = 0.1
bulge.effective_radius = 0.8

mass = af.Model(al.mp.SphIsothermal)
mass.centre = (0.0, 0.0)
mass.einstein_radius = 1.6

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.VoronoiNNMagnification(shape=(30, 30)),
    regularization=al.reg.ConstantSplit,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

instance = model.instance_from_prior_medians()

"""
__Paths__
"""
paths = af.DirectoryPaths(
    path_prefix=path.join("build", "visualizer"),
    name="imaging"
)

"""
__Position Likelihood__
"""
positions_likelihood = al.PositionsLHPenalty(positions=positions, threshold=1.0)

"""
__Analysis__ 
"""
analysis = al.AnalysisImaging(
    dataset=imaging, positions_likelihood=positions_likelihood
)

analysis.modify_before_fit(paths=paths, model=model)

analysis.visualize(
    paths=paths,
    instance=instance,
    during_analysis=False
)

"""
Finish.
"""
