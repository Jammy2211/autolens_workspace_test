"""
Viusalize: Point
================

This script performs an point model fit, where all images are output during visualization as .png and .fits
files.

This tests all visualization outputs in **PyAutoLens** for point data.
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

"""
__Dataset__
"""
dataset_label = "build"
dataset_type = "point_source"

dataset_path = path.join("dataset", dataset_label, dataset_type)

data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=0.05
)

dataset = al.from_json(
    file_path=path.join(dataset_path, "point_dataset.json"),
)

"""
__Model__
"""

mass = af.Model(al.mp.Isothermal)
mass.centre = (0.0, 0.0)
mass.ell_comps = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
mass.einstein_radius = 1.6

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)
source = af.Model(al.Galaxy, redshift=1.0, point_0=al.ps.Point)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__PointSolver__
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)


"""
__Search__
"""
search = af.DynestyStatic(
    path_prefix=path.join("build", "visualizer", "point_source"),
    name="test",
    unique_tag="test",
    nlive=50,
    number_of_cores=1,
)

"""
__Analysis__
"""
analysis = al.AnalysisPoint(dataset=dataset, solver=solver)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)


"""
Finish.
"""
