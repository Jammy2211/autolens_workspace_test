"""
__PROFILING: Plots__

This script creates plots for all `Imaging` profiling scripts performed by PyAutoLens.
"""
import autolens as al
import json
import os
import numpy as np
from os import path
import matplotlib.pyplot as plt

"""
The path containing all profiling results to be plotted is in a folder with the PyAutoLens version number.
"""
profiling_name = "inversion_voronoi_magnification_lens_subtract_high"

profiling_path = os.path.join("profiling", "times", al.__version__, profiling_name)

"""
The path where the profiling graphs created by this script are output, which is again a folder with the PyAutoLens 
version number.
"""
graph_path = os.path.join("profiling", "graphs", al.__version__, profiling_name)

if not os.path.exists(graph_path):
    os.makedirs(graph_path)

"""
Plots a bar chart of the deflection angle run-times from a deflections profiling dict.
"""


def bar_deflection_profiles(
    profiling_dict, fit_time, info_dict, file_path, filename, color="b"
):

    plt.figure(figsize=(18, 14))

    barlist = plt.barh(
        list(profiling_dict.keys()), list(profiling_dict.values()), color=color
    )

    [barlist[index].set_color("yellow") for index in range(0, 3)]
    [barlist[index].set_color("r") for index in range(3, 6)]
    [barlist[index].set_color("g") for index in range(6, 11)]

    colors = {
        "Lens Light": "yellow",
        "Lensing Calculations": "red",
        "Pixelization / Gridding": "green",
        "Linear Algebra": "blue",
    }

    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels, fontsize=20)

    if "hst" in filename:
        title = f"Voronoi Magnification Inversion HST (total time = {np.round(fit_time, 2)})"

    plt.yticks(fontsize=16)
    plt.xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], fontsize=20)
    plt.xlabel("Run Time (seconds)", fontsize=30)
    plt.title(title, fontsize=26)

    plt.text(1, 20.5, f'Image Sub-Pixels = {info_dict["image_pixels"]}', fontsize=20)
    plt.text(1, 19.5, f'Source Pixels = {info_dict["source_pixels"]}', fontsize=20)
    plt.text(1, 18.5, f'Sub Size = {info_dict["sub_size"]}', fontsize=20)
    plt.text(1, 17.5, f'Mask Radius = {info_dict["mask_radius"]}"', fontsize=20)
    plt.text(1, 16.5, f'PSF 2D Shape = {info_dict["psf_shape_2d"]}', fontsize=20)

    plt.savefig(path.join(file_path, f"{filename}.png"), bbox_inches="tight")
    plt.close()


"""
Load the `Inversion` profiling run times of the `VoronoiMagnification` pixelization.
"""
file_path = path.join(profiling_path, "hst_profiling_dict.json")
with open(file_path, "r") as f:
    profiles_dict = json.load(f)

"""
Load the total run time of the `VoronoiMagnification` pixelization.
"""
file_path = path.join(profiling_path, "hst_fit_time.json")
with open(file_path, "r") as f:
    fit_time = json.load(f)


"""
Load the `info_dict` of the `VoronoiMagnification` pixelization run.
"""
file_path = path.join(profiling_path, "hst_info.json")
with open(file_path, "r") as f:
    info_dict = json.load(f)

bar_deflection_profiles(
    profiling_dict=profiles_dict,
    fit_time=fit_time,
    info_dict=info_dict,
    file_path=graph_path,
    filename="hst_profiling",
)
