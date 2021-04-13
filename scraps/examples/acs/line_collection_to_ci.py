import numpy as np
import os
from os import path

import pickle
import autocti as ac

"""Load the HST ACS dataset"""
dataset_path = path.join("dataset", "examples", "acs", "lines")
dataset_name = "jc0a01h8q"

lines = ac.LineCollection()
lines.load(filename=path.join(dataset_path, dataset_name, ".pickle"))

imaging_ci_list = []

delta_function_index = 9

for i in range(lines.n_lines):

    pattern_ci = ac.ci.PatternCIUniform(
        regions=[(delta_function_index, delta_function_index + 1, 0, 1)],
        normalization=lines.lines[i].grid_plot[delta_function_index],
    )

    """
    The location attribute is the location of the warm pixel. Each cut-out line is 18 pixels, including 9 pixels before 
    the peak flux in the warm pixel itself and 8 pixels after which include its trail.
    
    We need to compute the readout_offset of this warm pixel image, which is the distance from the first
    pixel in the image (e.g. the pixel which is 9 pixel in front of the warm pixel itself) to the serial
    register. Thus, we subtract 9 from the warm pixel location.
    """

    readout_offsets = (
        lines.lines[i].location[0] - delta_function_index,
        lines.lines[i].location[1],
    )

    image = ac.ci.CIFrame.manual(
        array=lines.lines[i].grid_plot[:, None],
        pixel_scales=0.05,
        pattern_ci=pattern_ci,
        exposure_info=ac.ExposureInfo(readout_offsets=readout_offsets),
    )

    noise_map = ac.ci.CIFrame.manual(
        # array=lines.lines[i].noise_map[:,None],
        array=np.ones(shape=image.shape_native),
        pixel_scales=0.05,
        pattern_ci=pattern_ci,
        exposure_info=ac.ExposureInfo(readout_offsets=readout_offsets),
    )

    pre_cti_ci = pattern_ci.pre_cti_ci_from(
        shape_native=image.shape_native, pixel_scales=image.pixel_scales
    )

    imaging_ci_list.append(
        ac.ci.ImagingCI(
            image=image,
            noise_map=noise_map,
            pre_cti_ci=pre_cti_ci,
            name=lines.lines[i].name,
        )
    )

"""Serialize the `ImagingCI`'s so we can load them in the CTI model fitting scripts."""

output_path = path.join("dataset", "examples", "acs", "lines")

if not os.path.exists(output_path):
    os.mkdir(output_path)

with open(path.join(output_path, dataset_name, "imaging_ci_list.pickle"), "wb") as f:
    pickle.dump(imaging_ci_list, f)
