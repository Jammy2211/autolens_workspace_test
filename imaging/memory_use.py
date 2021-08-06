# %%
"""
__PROFILING: Interferometer Memory Use__

This simple tool estimates how much memory will be used when fitting an `Interferometer` data.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import autolens as al

"""
The total number of visibilties in your dataset, which drives memory use.
"""
psf_size_slim = 21
total_image_pixels = 500000

"""
The real-space mask used by your analysis.
"""
mask = al.Mask2D.circular(
    shape_native=(256, 256), pixel_scales=0.05, sub_size=1, radius=3.0
)

real_space_pixels = mask.sub_pixels_in_mask

"""
If you use an `Inversion` the number of source-pixels in your fit (2000 is a good rough estimate).
"""
source_pixels = 2000

shape_data = total_image_pixels * (3 * psf_size_slim ** 2)


print("Data Memory Use (GB) = " + str(shape_data * 8e-9))
print()
