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
import autolens.plot as aplt

"""
__Dataset__
"""
dataset_name = "mass_sie__source_point__0"
dataset_path = path.join("dataset", "point_source", dataset_name)

image_2d = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "image.fits"), pixel_scales=0.05
)

point_dict = al.PointDict.from_json(
    file_path=path.join(dataset_path, "point_dict.json")
)

"""
__Model__
"""

mass = af.Model(al.mp.EllIsothermal)
mass.centre = (0.0, 0.0)
mass.elliptical_comps = al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0)
mass.einstein_radius = 1.6

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)
source = af.Model(al.Galaxy, redshift=1.0, point_0=al.ps.Point)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__PointSolver__
"""
grid_2d = al.Grid2D.uniform(
    shape_native=image_2d.shape_native, pixel_scales=image_2d.pixel_scales
)

point_solver = al.PointSolver(grid=grid_2d, pixel_scale_precision=0.025)

"""
__Search__
"""
search = af.DynestyStatic(
    path_prefix=path.join("visualizer"),
    name="point_source",
    unique_tag=dataset_name,
    nlive=50,
    number_of_cores=1,
)

"""
__Analysis__
"""
analysis = al.AnalysisPoint(point_dict=point_dict, solver=point_solver)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)


"""
Finish.
"""
