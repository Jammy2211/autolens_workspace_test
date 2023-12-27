"""
__PROFILING: Inversion Voronoi__

This profiling script times how long it takes to fit `Imaging` data with a `Voronoi` pixelization for
datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoLens** log likelihood function.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import json
import numpy as np
import time

import autolens as al

"""
The path all profiling results are output.
"""
file_path = os.path.join(
    "imaging",
    "profiling",
    "times",
    al.__version__,
    "inversion_voronoi_magnification__w_tilde",
)

"""
The number of repeats used to estimate the `Inversion` run time.
"""
repeats = conf.instance["general"]["profiling"]["repeats"]
repeats = 1
print("Number of repeats = " + str(repeats))
print()

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 1
pixel_scales = (0.05, 0.05)
mask_radius = 3.5
mesh_shape_2d = (57, 57)

print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"pixelization shape = {mesh_shape_2d}")

"""
These settings control the run-time of the `Inversion` performed on the `Interferometer` data.
"""
transformer_class = al.TransformerNUFFT
use_linear_operators = False

"""
Set up the `Interferometer` dataset we fit. This includes the `real_space_mask` that the source galaxy's 
`Inversion` is evaluated using via mapping to Fourier space using the `Transformer`.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(800, 800),
    pixel_scales=pixel_scales,
    sub_size=sub_size,
    radius=mask_radius,
)

# mask = al.Mask2D.circular_annular(
#     shape_native=dataset.shape_native,
#     pixel_scales=dataset.pixel_scales,
#     sub_size=sub_size,
#     inner_radius=1.5,
#     outer_radius=2.5,
# )

"""
Load the strong lens dataset `mass_sie__source_sersic` `from .fits files.
"""
instrument = "sma"

dataset_path = path.join("dataset", "interferometer", "instruments", instrument)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

shape_vis = 200000
dataset.noise_map = 10000 + 10000j * np.ones(shape_vis)
dataset.uv_wavelengths = 10000.0 * np.random.rand(shape_vis, 2)

"""
__Numba Caching__

Call the dataset cached properties once to get all numba functions initialized.
"""
masked_dataset.convolver
del masked_dataset.__dict__["convolver"]

masked_dataset.w_tilde
del masked_dataset.__dict__["w_tilde"]

"""
__Profiling Dict__

Time how long each masked imaging cached variable takes to compute.
"""
caching_dict = {}

start = time.time()
for i in range(repeats):
    masked_dataset.convolver
    del masked_dataset.__dict__["convolver"]
time_calc = (time.time() - start) / repeats

caching_dict["convolver"] = time_calc

start = time.time()
for i in range(repeats):
    masked_dataset.w_tilde
    print(masked_dataset.w_tilde.curvature_preload.shape)
    del masked_dataset.__dict__["w_tilde"]
time_calc = (time.time() - start) / repeats

caching_dict["w_tilde"] = time_calc

"""
__Results__

These two numbers are the primary driver of run time. More pixels = longer run time.
"""

print(f"Dataset Caching run times for image type {instrument} \n")
print(f"Number of pixels = {masked_dataset.grid.shape_slim} \n")
print(f"Number of sub-pixels = {masked_dataset.grid.sub_shape_slim} \n")

"""
Print the profiling results of every step of the fit for command line output when running profiling scripts.
"""
for key, value in caching_dict.items():
    print(key, value)

"""
__Output__

Output the profiling run times as a dictionary so they can be used in `profiling/graphs.py` to create graphs of the
profile run times.

This is stored in a folder using the **PyAutoLens** version number so that profiling run times can be tracked through
**PyAutoLens** development.
"""
if not os.path.exists(file_path):
    os.makedirs(file_path)

filename = f"{instrument}_caching_dict.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(caching_dict, outfile)

"""
The `info_dict` contains all the key information of the analysis which describes its run times.
"""
info_dict = {}
info_dict["repeats"] = repeats
info_dict["image_pixels"] = masked_dataset.grid.sub_shape_slim
info_dict["sub_size"] = sub_size
info_dict["mask_radius"] = mask_radius
info_dict["psf_shape_2d"] = psf_shape_2d

print(info_dict)

with open(path.join(file_path, f"{instrument}_info.json"), "w") as outfile:
    json.dump(info_dict, outfile)
