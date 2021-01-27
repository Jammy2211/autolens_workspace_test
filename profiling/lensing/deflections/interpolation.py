# %%
"""
__PROFILING: DEFLECTIONS__ 

This example profiles interpolated deflection-angles, which computes the deflection angles of a `MassProfile` 
on a coarse lower resolution `interpolation grid_interpolate` and interpolates these values to the image`s native 
sub-grid_interpolate resolution.

The benefits of this are:

 - For `MassProfile`'s that require computationally expensive numerical integration, this reduces the number of
   integrals performed 100000`s to 1000`s, giving a potential speed up in run time of x100 or more!

The downsides of this are:

 - The interpolated deflection angles will be inaccurate to some level of precision, depending on the resolution
   of the interpolation grid_interpolate. This could lead to inaccurate and biased mass models.

The interpolation grid_interpolate is defined in terms of a pixel scale and it is automatically matched to the mask used in that
phase. A higher resolution grid_interpolate (i.e. lower pixel scale) will give more precise deflection angles, at the expense
of longer calculation times. In this example we will use an interpolation pixel scale of 0.05", which balances run-time
and precision.
"""

import time
import autolens as al

# %%
"""
We will compute the deflection angles within a circular `Mask2D`, which is the same object defining how
many deflection angles we compute when we fit a lens-model. Below, we have chosen the resolution and size
of this `Mask2D` to be representative of Hubble Space Telescope (HST) imaging.

If your data is lower resolution you can anticipate faster run times than this tool, and are free to edit the
values below to be representative of your dataset.
"""
# %%
mask = al.Mask2D.circular(
    shape_native=(301, 301), pixel_scales=0.05, sub_size=2, radius=3.0
)

grid = al.Grid2D.from_mask(mask=mask)
print(
    f"Number of (y,x) on grid where deflection angles are computed = {grid.sub_shape_slim}"
)

pixel_scales_interp = 0.05
grid_interpolate = al.Grid2DInterpolate.from_mask(
    mask=mask, pixel_scales_interp=pixel_scales_interp
)
print(
    f"Number of (y,x) on grid_interpolate where deflection angles are computed = {grid_interpolate.grid_interp.sub_shape_slim}"
)
print(f"Interpolation Pixel Scale = {pixel_scales_interp}\n")


def time_deflections_from_grid(mass_profile, grid):
    start = time.time()
    mass_profile.deflections_from_grid(grid=grid)
    return time.time() - start


"""
We compare the uninterpolated and interpolated calculations for a selction of profiles below, demonstrating how
a speed-up is achieved for certain profiles.
"""

mass_profile = al.mp.EllipticalCoredPowerLaw(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    einstein_radius=1.0,
    slope=2.0,
    core_radius=0.1,
)

calculation_time_grid = time_deflections_from_grid(mass_profile=mass_profile, grid=grid)
calculation_time_interpolate = time_deflections_from_grid(
    mass_profile=mass_profile, grid=grid_interpolate
)
print("EllipticalCoredPowerLaw time using grid = {}".format(calculation_time_grid))
print(
    "EllipticalCoredPowerLaw time using interpolation = {}".format(
        calculation_time_interpolate
    )
)

mass_profile = al.mp.EllipticalNFW(
    centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), kappa_s=0.1, scale_radius=10.0
)

calculation_time_grid = time_deflections_from_grid(mass_profile=mass_profile, grid=grid)
calculation_time_interpolate = time_deflections_from_grid(
    mass_profile=mass_profile, grid=grid_interpolate
)
print("EllipticalNFW time using grid = {}".format(calculation_time_grid))
print(
    "EllipticalNFW time using interpolation = {}".format(calculation_time_interpolate)
)

mass_profile = al.mp.EllipticalSersic(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
    mass_to_light_ratio=1.0,
)

calculation_time_grid = time_deflections_from_grid(mass_profile=mass_profile, grid=grid)
calculation_time_interpolate = time_deflections_from_grid(
    mass_profile=mass_profile, grid=grid_interpolate
)
print("EllipticalSersic time using grid = {}".format(calculation_time_grid))
print(
    "EllipticalSersic time using interpolation = {}".format(
        calculation_time_interpolate
    )
)

mass_profile = al.mp.EllipticalSersicRadialGradient(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
    mass_to_light_ratio=1.0,
    mass_to_light_gradient=-1.0,
)

calculation_time_grid = time_deflections_from_grid(mass_profile=mass_profile, grid=grid)
calculation_time_interpolate = time_deflections_from_grid(
    mass_profile=mass_profile, grid=grid_interpolate
)
print(
    "EllipticalSersicRadialGradient time using grid = {}".format(calculation_time_grid)
)
print(
    "EllipticalSersicRadialGradient time using interpolation = {}".format(
        calculation_time_interpolate
    )
)
