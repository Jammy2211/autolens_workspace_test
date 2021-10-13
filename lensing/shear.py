import numpy as np

import autoarray as aa
import autolens as al

grid = al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.5)


class SphIsothermal(al.mp.SphIsothermal):
    def __init__(self, centre=(0.0, 0.0), einstein_radius: float = 1.0, slope=2.0):

        super().__init__(centre=centre, einstein_radius=einstein_radius)

        self.slope = slope

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_from(self, grid):
        grid_radii = self.grid_to_grid_radii(grid=grid)

        return ((3 - self.slope) / 2.0) * (grid_radii / self.einstein_radius) ** (
            1.0 - self.slope
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_from(self, grid):
        grid_radii = self.grid_to_grid_radii(grid=grid)

        return self.grid_to_grid_cartesian(
            grid=grid,
            radius=self.einstein_radius
            * (grid_radii / self.einstein_radius) ** (2.0 - self.slope),
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def shear_2d_from(self, grid):

        grid_radii = self.grid_to_grid_radii(grid=grid)

        radius = ((self.slope - 1.0) / 2.0) * (grid_radii / self.einstein_radius) ** (
            1.0 - self.slope
        )

        grid_thetas = np.arctan2(grid[:, 0], grid[:, 1])
        grid_thetas -= np.pi / 4.0

        theta_coordinate_to_profile = np.add(grid_thetas, -self.phi_radians)

        cos_theta, sin_theta = (
            np.cos(theta_coordinate_to_profile),
            np.sin(theta_coordinate_to_profile),
        )

        return np.multiply(radius[:, None], np.vstack((sin_theta, cos_theta)).T)

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def shear_2d_from_2(self, grid):
        grid_radii = self.grid_to_grid_radii(grid=grid)

        return self.grid_to_grid_cartesian(
            grid=grid,
            radius=((self.slope - 1.0) / 2.0)
            * (grid_radii / self.einstein_radius) ** (1.0 - self.slope),
        )

    def shear_magnitudes_from_grid(self, grid):
        shear_2d = self.shear_2d_from(grid=grid)

        return (shear_2d[:, 0] ** 2.0 + shear_2d[:, 1] ** 2.0) ** 0.5

    def shear_angles_from_grid(self, grid):
        shear_2d = self.shear_2d_from_2(grid=grid)

        return np.arctan2(shear_2d[:, 0], shear_2d[:, 1]) * 180.0 / np.pi / 2.0


sis = SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0, slope=2.0)

# point = al.mp.PointMass(centre=(0.0, 0.0), einstein_radius=2.0)

sis_al = al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

# print(sis.shear_magnitudes_from_grid(grid=grid)[0:5])
# print(sis_al.shear_via_hessian_from(grid=grid)[0:5])

print(sis.shear_angles_from_grid(grid=grid)[0:5])
print(sis.shear_angles_via_hessian_from(grid=grid)[0:5])
stop

print(
    sis.shear_angles_from_grid(grid=grid)[0] - sis.shear_angles_from_grid(grid=grid)[1]
)
print(
    (
        sis.shear_angles_via_hessian_from(grid=grid)[0]
        - sis.shear_angles_via_hessian_from(grid=grid)[1]
    )
    / 2.0
)

# print()

# print(shear_y_via_hessian[0:5])
# print(shear_x_via_hessian[0:5])
#
# shear_y_via_jacobian, shear_x_via_jacobian = point.shear_yx_via_jacobian_from(grid=grid)
# print(shear_y_via_jacobian[0:5])
# print(shear_x_via_jacobian[0:5])

stop2

sis = al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

shear_y_via_hessian, shear_x_via_hessian = sis.shear_yx_via_hessian_from(grid=grid)
shear_y_via_jacobian, shear_x_via_jacobian = sis.shear_yx_via_jacobian_from(grid=grid)

shear_via_hessian = sis.shear_via_hessian_from(grid=grid)
shear_via_jacobian = sis.shear_via_jacobian_from(grid=grid)

for i in range(shear_x_via_hessian.shape[0]):

    print(
        i,
        shear_y_via_hessian[i],
        shear_y_via_jacobian[i],
        shear_x_via_hessian[i],
        shear_x_via_jacobian[i],
        shear_via_hessian[i],
        shear_via_jacobian[i],
    )
