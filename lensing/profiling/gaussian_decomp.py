"""
__PROFILING: DEFLECTIONS GAUSSIAN DECOMP__

This profiling script times how long deflection angles take to compute for the gaussian decomposition method
in **PyAutoLens**.

It extracts the functionality in the `mass_profiles` module into a standalone script that makes it straight forward
to profile and edit for optimization.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import autolens as al
import time
import numpy as np

"""
We will compute the deflection angles within a circular `Mask2D`, which is the same object defining how
many deflection angles we compute when we fit a lens-model. Below, we have chosen the resolution and size
of this `Mask2D` to be representative of Hubble Space Telescope (HST) imaging.
"""
mask = al.Mask2D.circular(
    shape_native=(301, 301), pixel_scales=0.05, sub_size=4, radius=3.5
)

grid = al.Grid2D.from_mask(mask=mask)

"""
The number of pixels represents the number of deflection angle calculations that are performed and may therefore be
used to scale the profiling times to higher or lower resolution datasets.
"""
pixels = grid.sub_shape_slim

print(
    f"Number of (y,x) pixels on grid where deflection angles are computed = {pixels})\n"
)

"""
The number of repeats used to estimate the deflection angle calculation of every `MassProfile`.
"""
repeats = 1

"""
The gaussian decomposition method is used on multiple mass profiles, in this example we use it for the `EllSersic`
mass model. We do not use the class found in the `mass_profiles` module and write the individual parameters below.
"""
centre = (0.0, 0.0)
elliptical_comps = (0.0, 0.0)
intensity = 1.0
effective_radius = 0.5
sersic_index = 4.0
mass_to_light_ratio = 2.0

axis_ratio, angle = al.convert.axis_ratio_and_phi_from(elliptical_comps=elliptical_comps)

sersic_constant = (
    (2 * sersic_index)
    - (1.0 / 3.0)
    + (4.0 / (405.0 * sersic_index))
    + (46.0 / (25515.0 * sersic_index ** 2))
    + (131.0 / (1148175.0 * sersic_index ** 3))
    - (2194697.0 / (30690717750.0 * sersic_index ** 4))
)

radii_min = effective_radius / 100.0
radii_max = effective_radius * 20.0


def sersic_2d(r):
    return (
        mass_to_light_ratio
        * intensity
        * np.exp(
            -sersic_constant * (((r / effective_radius) ** (1.0 / sersic_index)) - 1.0)
        )
    )


import gaussian_decomp_util

"""
This is the function we want to speed up.
"""
start = time.time()

for i in range(repeats):

    deflections = gaussian_decomp_util.deflections_2d_from_grid_via_gaussians(
        func=sersic_2d,
        axis_ratio=axis_ratio,
        angle_profile=angle,
        grid=grid,
        radii_min=radii_min,
        radii_max=radii_max,
        func_terms=28,
        func_gaussians=20,
        sigmas_factor=np.sqrt(axis_ratio)
    )

time_util = (time.time() - start) / repeats
print(f"Time for edited function = {time_util}")

"""
Sanity check by comparing to autolens source code:
"""
start = time.time()

for i in range(repeats):

    sersic = al.mp.SphSersic(
        centre=centre,
      #  elliptical_comps=elliptical_comps,
        intensity=intensity,
        effective_radius=effective_radius,
        sersic_index=sersic_index,
        mass_to_light_ratio=mass_to_light_ratio
    )

    deflections_sersic = sersic.deflections_2d_from_grid(grid=grid)

time_source = (time.time() - start) / repeats
print(f"Time for source code function = {time_source}")

print()
print("Sanity Check:")
print(np.max(deflections-deflections_sersic))