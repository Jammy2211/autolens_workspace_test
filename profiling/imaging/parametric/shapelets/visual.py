"""
__PROFILING: Parametric__

This profiling script times how long it takes to fit `Imaging` data with a parametric `Sersic` lens galaxy bulge
and source galaxy bulge, after lensing by a mass profile.

This represents the time taken by a single iteration of the **PyAutoLens** log likelihood function.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import numpy as np
import time
import json

import autofit as af
import autolens as al
import autolens.plot as aplt


"""
The path all profiling results are output.
"""
profiling_path = path.dirname(path.realpath(__file__))

file_path = os.path.join(profiling_path, "times", al.__version__)

"""
The number of repeats used to estimate the run time.
"""
repeats = conf.instance["general"]["profiling"]["repeats"]
print("Number of repeats = " + str(repeats))
print()

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
grid_class = al.Grid2D
sub_size = 1
mask_radius = 3.5
psf_shape_2d = (21, 21)

print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"psf shape = {psf_shape_2d}")

settings_inversion = al.SettingsInversion(use_positive_only_solver=False)

"""
The source galaxy whose `Voronoi` `Pixelization` fits the data.
"""
total_xy = 5

shapelets_dict = {}

for x in range(total_xy):
    for y in range(total_xy):
        shapelet = al.lp.ShapeletCartesianSph(n_y=y, n_x=x, centre=(0.0, 0.0), beta=1.0)

        shapelets_dict[f"shapelet_{x}_{y}"] = shapelet

# total_n = 5
# total_m = sum(range(2, total_n + 1)) + 1
#
# shapelets_bulge_list = []
# shapelets_dict = {}
#
# n_count = 1
# m_count = -1
#
# for i in range(total_n + total_m):
#
#     shapelet = al.lp.ShapeletPolar(
#         n=n_count,
#         m=m_count,
#         centre=(0.1, 0.1),
#         ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
#         intensity=1.0,
#         beta=1.0
#     )
#
#     shapelets_dict[f"shapelet_{n_count}_{m_count}"] = shapelet
#
# #    shapelets_bulge_list.append(shapelet)
#
#     m_count += 2
#
#     if m_count > n_count:
#         n_count += 1
#         m_count = -n_count

source = al.Galaxy(
    redshift=1.0,
    **shapelets_dict,
)

output = aplt.Output(
    path=profiling_path,
    filename="shapelets",
    format="png",
)

plotter = aplt.GalaxyPlotter(
    galaxy=source,
    grid=al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05),
    mat_plot_2d=aplt.MatPlot2D(output=output),
)

light_profile_plotters = [
    plotter.light_profile_plotter_from(light_profile)
    for light_profile in source.cls_list_from(cls=al.LightProfile)
]

n_count = 1
m_count = -1

for i, lpp in enumerate(light_profile_plotters):
    lpp.set_title(f"shapelet_{n_count}_{m_count}")

    print(f"n = {n_count}, m = {m_count}")

    m_count += 2

    if m_count > n_count:
        n_count += 1
        m_count = -n_count

plotter.subplot_of_plotters_figure(plotter_list=light_profile_plotters, name="image")
