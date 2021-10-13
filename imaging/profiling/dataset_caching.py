"""
__PROFILING: Inversion VoronoiMagnification__

This profiling script times how long it takes to fit `Imaging` data with a `VoronoiMagnification` pixelization for
datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoLens** log likelihood function.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import time
import json

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
sub_size = 4
mask_radius = 3.5
psf_shape_2d = (21, 21)

print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"psf shape = {psf_shape_2d}")

"""
The simulated data comes at five resolution corresponding to five telescopes:

vro: pixel_scale = 0.2", fastest run times.
euclid: pixel_scale = 0.1", fast run times
hst: pixel_scale = 0.05", normal run times, represents the type of data we do most our fitting on currently.
hst_up: pixel_scale = 0.03", slow run times.
ao: pixel_scale = 0.01", very slow :(
"""
# instrument = "vro"
# instrument = "euclid"
instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

pixel_scales_dict = {"vro": 0.2, "euclid": 0.1, "hst": 0.05, "hst_up": 0.03, "ao": 0.01}
pixel_scale = pixel_scales_dict[instrument]

"""
Load the dataset for this instrument / resolution.
"""
dataset_path = path.join("dataset", "imaging", "instruments", instrument)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scale,
)

"""
Apply the 2D mask, which for the settings above is representative of the masks we typically use to model strong lenses.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    sub_size=sub_size,
    radius=mask_radius,
)

# mask = al.Mask2D.circular_annular(
#     shape_native=imaging.shape_native,
#     pixel_scales=imaging.pixel_scales,
#     sub_size=sub_size,
#     inner_radius=1.5,
#     outer_radius=2.5,
# )

masked_imaging = imaging.apply_mask(mask=mask)

masked_imaging = masked_imaging.apply_settings(
    settings=al.SettingsImaging(sub_size=sub_size)
)

"""
__Numba Caching__

Call the dataset cached properties once to get all numba functions initialized.
"""
masked_imaging.convolver
del masked_imaging.__dict__["convolver"]

masked_imaging.w_tilde
del masked_imaging.__dict__["w_tilde"]

"""
__Profiling Dict__

Time how long each masked imaging cached variable takes to compute.
"""
caching_dict = {}

start = time.time()
for i in range(repeats):
    masked_imaging.convolver
    del masked_imaging.__dict__["convolver"]
time_calc = (time.time() - start) / repeats

caching_dict["convolver"] = time_calc

start = time.time()
for i in range(repeats):
    masked_imaging.w_tilde
    print(masked_imaging.w_tilde.curvature_preload.shape)
    del masked_imaging.__dict__["w_tilde"]
time_calc = (time.time() - start) / repeats

caching_dict["w_tilde"] = time_calc

"""
__Results__

These two numbers are the primary driver of run time. More pixels = longer run time.
"""

print(f"Dataset Caching run times for image type {instrument} \n")
print(f"Number of pixels = {masked_imaging.grid.shape_slim} \n")
print(f"Number of sub-pixels = {masked_imaging.grid.sub_shape_slim} \n")

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
info_dict["image_pixels"] = masked_imaging.grid.sub_shape_slim
info_dict["sub_size"] = sub_size
info_dict["mask_radius"] = mask_radius
info_dict["psf_shape_2d"] = psf_shape_2d

print(info_dict)

with open(path.join(file_path, f"{instrument}_info.json"), "w") as outfile:
    json.dump(info_dict, outfile)
