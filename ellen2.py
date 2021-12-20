import autolens as al
import autolens.plot as aplt

"""
We compute lensing quantities such as convergence and shear using a 2D grid of (y,x) coordinates.

For the grid below, the y coordinates are 1.0", 3.0" and x coordinates 2.0", 4.0".
"""
grid = al.Grid2DIrregular(grid=[[1.0, 2.0], [3.0, 4.0]])

"""
Set up an `EllNFW` mass profile which we will use to compute its lensing derived quantities.
"""
nfw = al.mp.EllNFW(
    centre=(0.0, 0.0),
    elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=45.0),
    kappa_s=0.1,
    scale_radius=30.0,
)

"""
The function `convergence_2d_from` gives us the convergence, computed analytically.
"""
convergence = nfw.convergence_2d_from(grid=grid)
print(convergence)

"""
The shear of a mass profile is not computed analytically. 

It is derived using the deflection angle map, via the lensing Hessian and it is returned as a vector field:
"""
shear_yx = nfw.shear_yx_2d_via_hessian_from(grid=grid)

shear_y = shear_yx[0]
shear_x = shear_yx[1]

print(shear_y)
print(shear_x)

import metpy
