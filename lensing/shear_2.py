import numpy as np

import autoarray as aa
import autolens as al

grid = al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.5)

shear = al.mp.ExternalShear(elliptical_comps=(0.2, 0.4))

shear_y_via_hessian, shear_x_via_hessian = shear.shear_yx_2d_via_hessian_from(grid=grid)
print("")
shear_y_via_jacobian, shear_x_via_jacobian = shear.shear_yx_2d_via_jacobian_from(
    grid=grid
)


for i in range(shear_x_via_hessian.shape[0]):

    print(
        i,
        shear_y_via_hessian[i],
        shear_y_via_jacobian[i],
        shear_x_via_hessian[i],
        shear_x_via_jacobian[i],
    )
