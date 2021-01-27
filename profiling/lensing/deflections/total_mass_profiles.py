# %%
"""
__PROFILING: DEFLECTIONS__ 

This profiling script times how long deflection angles take to compute for different `MassProfile`'s
in **PyAutoLens**. 
"""

# %%
import autolens as al
import time

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
    f"Number of (y,x) on grid where deflection angles are computed = {grid.sub_shape_slim})\n"
)

# %%
"""
The function below times the deflection angle calculation on an input `MassProfile`.
"""

# %%
def time_deflections_from_grid(mass_profile):
    start = time.time()
    mass_profile.deflections_from_grid(grid=grid)
    return time.time() - start


mass_profile = al.mp.EllipticalIsothermal(
    centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=1.0
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("EllipticalIsothermal calculation_time = {}".format(calculation_time))

mass_profile = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("SphericalIsothermal calculation_time = {}".format(calculation_time))

mass_profile = al.mp.EllipticalPowerLaw(
    centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=1.0, slope=1.5
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("EllipticalPowerLaw (slope = 1.5) calculation_time = {}".format(calculation_time))

mass_profile = al.mp.SphericalPowerLaw(
    centre=(0.0, 0.0), einstein_radius=1.0, slope=1.5
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("SphericalPowerLaw (slope = 1.5) calculation_time = {}".format(calculation_time))


mass_profile = al.mp.EllipticalPowerLaw(
    centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=1.0, slope=2.5
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("EllipticalPowerLaw (slope = 2.5) calculation_time = {}".format(calculation_time))

mass_profile = al.mp.SphericalPowerLaw(
    centre=(0.0, 0.0), einstein_radius=1.0, slope=2.5
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("SphericalPowerLaw (slope = 2.5) calculation_time = {}".format(calculation_time))

mass_profile = al.mp.EllipticalBrokenPowerLaw(
    centre=(0.0, 0.0),
    elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=30.0),
    einstein_radius=3.0,
    inner_slope=1.5,
    outer_slope=2.5,
    break_radius=0.2,
)

calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("EllipticalBrokenPowerLaw calculation_time = {}".format(calculation_time))

mass_profile = al.mp.SphericalBrokenPowerLaw(
    centre=(0.0, 0.0),
    einstein_radius=3.0,
    inner_slope=1.5,
    outer_slope=2.5,
    break_radius=0.2,
)

calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("SphericalBrokenPowerLaw calculation_time = {}".format(calculation_time))

mass_profile = al.mp.EllipticalCoredPowerLaw(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    einstein_radius=1.0,
    slope=2.0,
    core_radius=0.1,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("EllipticalCoredPowerLaw calculation_time = {}".format(calculation_time))

mass_profile = al.mp.SphericalCoredPowerLaw(
    centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0, core_radius=0.1
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("SphericalCoredPowerLaw calculation_time = {}".format(calculation_time))
