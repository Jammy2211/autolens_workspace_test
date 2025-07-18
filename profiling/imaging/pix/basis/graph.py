"""
__PROFILING: Plots__

This script creates plots for all `Imaging` profiling scripts performed by PyAutoLens.
"""

import os
from os import path

import autolens as al
import json
import numpy as np
from os import path
import matplotlib.pyplot as plt

# instrument = "vro"
instrument = "euclid"
# instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"


"""
The path containing all profiling results and graphs for this setup.
"""
profiling_path = path.dirname(path.realpath(__file__))

"""
The path containing all profiling results to be plotted is in a folder with the PyAutoLens version number.
"""
times_path = os.path.join(profiling_path, "times", al.__version__)

"""
The path where the profiling graphs created by this script are output, which is again a folder with the PyAutoLens 
version number.
"""
graph_path = os.path.join(profiling_path, "graphs", al.__version__)


if not os.path.exists(graph_path):
    os.makedirs(graph_path)

"""
Plots a bar chart of the deflection angle run-times from a deflections profiling dict.
"""


def bar_deflection_profiles(
    run_time_dict, fit_time, info_dict, file_path, filename, color="b"
):
    plt.figure(figsize=(14, 18))

    barlist = plt.barh(
        list(run_time_dict.keys()), list(run_time_dict.values()), color=color
    )

    [barlist[index].set_color("red") for index in range(0, 1)]
    [barlist[index].set_color("yellow") for index in range(1, 2)]
    [barlist[index].set_color("green") for index in range(2, 3)]
    [barlist[index].set_color("m") for index in range(3, 4)]

    colors = {
        "Ray Tracing": "yellow",
        "Source (Gaussian)": "green",
        "2D Convolution": "magenta",
        "Linear Algebra": "blue",
        "Source (Voronoi)": "red",
    }

    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels, fontsize=24)

    if instrument in filename:
        title = f"""
            PyAutoLens Analsyis Profiling 
            Simple Source Time = {np.round(fit_time, 3)} s
            Complex Source Time = {np.round(1.6530975843904, 3)} s
            """

    plt.yticks(fontsize=22)
    plt.xticks(fontsize=26)
    plt.xlabel("Run Time (seconds)", fontsize=30)
    plt.title(title, fontsize=34)

    # plt.text(1, 20.5, f'Image Sub-Pixels = {info_dict["image_pixels"]}', fontsize=20)
    # plt.text(1, 19.5, f'Source Pixels = {info_dict["source_pixels"]}', fontsize=20)
    # plt.text(1, 18.5, f'Sub Size = {info_dict["sub_size"]}', fontsize=20)
    # plt.text(1, 17.5, f'Mask Radius = {info_dict["mask_radius"]}"', fontsize=20)
    # plt.text(1, 16.5, f'PSF 2D Shape = {info_dict["psf_shape_2d"]}', fontsize=20)

    plt.savefig(path.join(file_path, f"{filename}.png"), bbox_inches="tight")
    plt.close()


"""
Load the `Inversion` profiling run times of the `Voronoi` pixelization.
"""
file_path = path.join(times_path, f"{instrument}_run_time_dict.json")
with open(file_path, "r") as f:
    profiles_dict = json.load(f)

"""
Load the total run time of the `Voronoi` pixelization.
"""
file_path = path.join(times_path, f"{instrument}_fit_time.json")
with open(file_path, "r") as f:
    fit_time = json.load(f)


"""
Load the `info_dict` of the `Voronoi` pixelization run.
"""
file_path = path.join(times_path, f"{instrument}_info.json")
with open(file_path, "r") as f:
    info_dict = json.load(f)

bar_deflection_profiles(
    run_time_dict=profiles_dict,
    fit_time=fit_time,
    info_dict=info_dict,
    file_path=graph_path,
    filename=f"{instrument}_profiling",
)
