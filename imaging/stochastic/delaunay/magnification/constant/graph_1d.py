"""
__PROFILING: Plots__

This script creates plots for all `Imaging` profiling scripts performed by PyAutoLens.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "fit"))

import autolens as al
import json
import numpy as np
from os import path
import matplotlib.pyplot as plt

"""
The path containing all profiling results and graphs for this setup.
"""
stochastic_path = path.relpath(path.dirname(__file__))

"""
The path containing all profiling results to be plotted is in a folder with the PyAutoLens version number.
"""
samples_path = os.path.join(stochastic_path, "samples_1d_70x70")

"""
The path where the profiling graphs created by this script are output, which is again a folder with the PyAutoLens 
version number.
"""
graph_path = os.path.join(stochastic_path, "graphs")

if not os.path.exists(graph_path):
    os.makedirs(graph_path)

"""
Plots a subplot of the different likelihood samples.
"""


def stochastic_subplot(stochastic_dict, info_dict, file_path, filename):
    def plot_subplot(subplot_index, y, ylabel, flip_y=False, color="b"):

        if flip_y:
            y = [-1.0 * yval for yval in y]

        plt.subplot(3, 2, subplot_index)
        plt.plot(stochastic_dict["slope_list"], y, color)
        plt.ylabel(ylabel, fontsize=10)
        plt.xlabel("Slope", fontsize=10)
        plt.yticks(fontsize=7)
        plt.xticks(fontsize=7)

    plt.figure(figsize=(18, 14))

    plot_subplot(
        subplot_index=1,
        y=stochastic_dict["chi_squared_list"],
        ylabel="Chi Squared",
        flip_y=True,
    )
    plot_subplot(
        subplot_index=2,
        y=stochastic_dict["regularization_term_list"],
        ylabel="Regularization Term",
        flip_y=True,
    )
    plot_subplot(
        subplot_index=3,
        y=stochastic_dict["log_det_curvature_reg_matrix_term_list"],
        ylabel="Log Det F + Lambda H Term",
        flip_y=True,
    )
    plot_subplot(
        subplot_index=4,
        y=stochastic_dict["log_det_regularization_matrix_term_list"],
        ylabel="Log Det Lambda H Term",
    )
    plot_subplot(
        subplot_index=5,
        y=stochastic_dict["noise_normalization_list"],
        ylabel="Noise Normalization Term",
        flip_y=True,
    )
    plot_subplot(
        subplot_index=6,
        y=stochastic_dict["figure_of_merit_list"],
        ylabel="Overall Log Evidence",
        color="r",
    )

    # plt.subplot(2, 4, 4)
    # plt.text(0.0, -0.2, f'Image Sub-Pixels = {info_dict["image_pixels"]}', fontsize=20)
    # plt.text(0.0, -0.3, f'Source Pixels = {info_dict["source_pixels"]}', fontsize=20)
    # plt.text(0.0, -0.4, f'Sub Size = {info_dict["sub_size"]}', fontsize=20)
    # plt.text(0.0, -0.5, f'Mask Radius = {info_dict["mask_radius"]}"', fontsize=20)

    plt.savefig(path.join(file_path, f"{filename}.png"), bbox_inches="tight")
    plt.close()


"""
Load the `Inversion` stochastic values.
"""
file_path = path.join(samples_path, "hst_stochastic_dict.json")
with open(file_path, "r") as f:
    stochastic_dict = json.load(f)

"""
Load the `info_dict` of the `DelaunayMagnification` mesh run.
"""
file_path = path.join(samples_path, "hst_info.json")
with open(file_path, "r") as f:
    info_dict = json.load(f)

print(graph_path)

stochastic_subplot(
    stochastic_dict=stochastic_dict,
    info_dict=info_dict,
    file_path=graph_path,
    filename="hst_stochastic_70x70",
)
