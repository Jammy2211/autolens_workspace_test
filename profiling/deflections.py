import os
from os import path
from skimage import measure

cwd = os.getcwd()

import autolens as al
import autolens.plot as aplt

import time

mask = al.Mask2D.circular(shape_native=(200, 200), pixel_scales=0.05, radius=3.0)

grid = al.Grid2D.from_mask(mask=mask)

"""
__SIS__
"""
mass = al.mp.IsothermalSph(
    centre=(0.0, 0.0),
    einstein_radius=2.0,
)
deflections = mass.deflections_yx_2d_from(grid=grid)


start = time.time()
deflections = mass.deflections_yx_2d_from(grid=grid)
print(f"SIS {time.time() - start}")


"""
__NFWSph__
"""
mass = al.mp.NFWSph(
    centre=(0.0, 0.0),
)
deflections = mass.deflections_yx_2d_from(grid=grid)

start = time.time()
deflections = mass.deflections_yx_2d_from(grid=grid)
print(f"NFWSph {time.time() - start}")


"""
__gNFWSph__
"""
mass = al.mp.gNFWSph(
    centre=(0.0, 0.0),
)
deflections = mass.deflections_yx_2d_from(grid=grid)

start = time.time()
deflections = mass.deflections_yx_2d_from(grid=grid)
print(f"gNFWSph {time.time() - start}")
