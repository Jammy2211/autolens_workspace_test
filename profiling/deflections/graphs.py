"""
__PROFILING: Graphs__

This script creates graphs for all `MassProfile` profiling scripts performed by PyAutoLens.
"""
import autolens as al
import json
import os
from os import path
import matplotlib.pyplot as plt

"""
The path containing all profiling results to be plotted is in a folder with the PyAutoLens version number.
"""
profiling_path = os.path.join("profiling", "times", al.__version__, "deflections")

"""
The path where the profiling graphs created by this script are output, which is again a folder with the PyAutoLens 
version number.
"""
graph_path = os.path.join("profiling", "graphs", al.__version__, "deflections")

if not os.path.exists(graph_path):
    os.makedirs(graph_path)

"""
Plots a bar chart of the deflection angle run-times from a deflections profiling dict.
"""


def bar_deflection_profiles(
    total_mass_profiles_dict,
    dark_mass_profiles_dict,
    stellar_mass_profiles_dict,
    file_path,
    filename,
):

    plt.figure(figsize=(18, 14))

    #  profiling_dict = total_mass_profiles_dict + dark_mass_profiles_dict + stellar_mass_profiles_dict

    plt.barh(
        list(total_mass_profiles_dict.keys()),
        list(total_mass_profiles_dict.values()),
        color="red",
    )
    plt.barh(
        list(stellar_mass_profiles_dict.keys()),
        list(stellar_mass_profiles_dict.values()),
        color="orange",
    )
    plt.barh(
        list(dark_mass_profiles_dict.keys()),
        list(dark_mass_profiles_dict.values()),
        color="k",
    )

    colors = {
        "Total Mass Profile": "red",
        "Stellar Mass Profile": "orange",
        "Dark Mass Profile": "k",
    }
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels, fontsize=20)

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=20)
    plt.xlabel("Run Time (seconds)", fontsize=20)
    pixels = filename.strip(".json")
    pixels = int(pixels.strip("pixels_"))
    plt.title(
        f"Deflection angle calculation run times for {pixels} sub-pixels", fontsize=26
    )
    plt.savefig(path.join(file_path, f"{filename}.png"), bbox_inches="tight")
    plt.close()


"""
Load the deflection angle calculation run times for the total mass profiles and output as a bar chart.
"""
filename_list = list(os.listdir(profiling_path))
filename = [
    filename for filename in filename_list if "total_mass_profiles" in filename
][0]
file_path = path.join(profiling_path, filename)

with open(file_path, "r") as f:
    total_mass_profiles_dict = json.load(f)


"""
Load the deflection angle calculation run times for the stellar mass profiles and output as a bar chart.
"""
filename_list = list(os.listdir(profiling_path))
filename = [
    filename for filename in filename_list if "stellar_mass_profiles" in filename
][0]
file_path = path.join(profiling_path, filename)

with open(file_path, "r") as f:
    stellar_mass_profiles_dict = json.load(f)

"""
Load the deflection angle calculation run times for the total mass profiles and output as a bar chart.
"""
filename_list = list(os.listdir(profiling_path))
filename = [filename for filename in filename_list if "dark_mass_profiles" in filename][
    0
]
file_path = path.join(profiling_path, filename)

with open(file_path, "r") as f:
    dark_mass_profiles_dict = json.load(f)


filename = filename.split("dark_mass_profiles__")[1]

bar_deflection_profiles(
    total_mass_profiles_dict=total_mass_profiles_dict,
    stellar_mass_profiles_dict=stellar_mass_profiles_dict,
    dark_mass_profiles_dict=dark_mass_profiles_dict,
    file_path=graph_path,
    filename=filename,
)
