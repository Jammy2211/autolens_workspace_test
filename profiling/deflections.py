import os
from os import path
from skimage import measure

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "shear"))

import autolens as al
import autolens.plot as aplt

import time

grid = al.Grid2D.uniform(shape_native=(20000, 20000), pixel_scales=1.0)

mass = al.mp.IsothermalSph(
    centre=(0.0, 0.0),
    einstein_radius=26.0,
)

"""
__Jacobian__
"""
start = time.time()
jacobian = mass.jacobian_from(grid=grid)
print(f"jacobian_from {time.time() - start}")

"""
__Convergence__
"""
start = time.time()
convergence = mass.convergence_2d_via_jacobian_from(grid=grid, jacobian=jacobian)
print(f"convergence_2d_via_jacobian_from {time.time() - start}")

"""
__Shear__
"""
start = time.time()
shear_yx = mass.shear_yx_2d_via_jacobian_from(grid=grid, jacobian=jacobian)
print(f"shear_yx_2d_via_jacobian_from {time.time() - start}")

"""
__Tangential Eigen Values__
"""
start = time.time()
tangential_eigen_values = al.Array2D(
    values=1 - convergence - shear_yx.magnitudes, mask=grid.mask
)
print(f"tangential_eigen_values {time.time() - start}")

"""
__Find Contours__
"""
start = time.time()
tangential_critical_curve_indices_list = measure.find_contours(
    tangential_eigen_values.native, 0
)
print(f"find_contours {time.time() - start}")


# mass.shear_yx_2d_via_hessian_from(grid=grid)

# end = time.time()
# print(end - start)
