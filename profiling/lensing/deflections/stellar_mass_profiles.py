"""
__PROFILING: DEFLECTIONS__ 

This profiling script times how long deflection angles take to compute for different `MassProfile`'s
in **PyAutoLens**. 
"""


import autolens as al
import time


"""
We will compute the deflection angles within a circular `Mask2D`, which is the same object defining how
many deflection angles we compute when we fit a lens-model. Below, we have chosen the resolution and size
of this `Mask2D` to be representative of Hubble Space Telescope (HST) imaging.

If your data is lower resolution you can anticipate faster run times than this tool, and are free to edit the
values below to be representative of your dataset.
"""


mask = al.Mask2D.circular(
    shape_native=(301, 301), pixel_scales=0.05, sub_size=2, radius=3.0
)

grid = al.Grid2D.from_mask(mask=mask)

print(
    f"Number of (y,x) on grid where deflection angles are computed = {grid.sub_shape_slim})\n"
)


"""
The function below times the deflection angle calculation on an input `MassProfile`.
"""

repeats = 10


def time_deflections_from_grid(mass_profile):
    start = time.time()
    for i in range(repeats):
        mass_profile.deflections_from_grid(grid=grid)
    return (time.time() - start) / repeats


mass_profile = al.mp.EllipticalChameleon(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    core_radius_0=0.05,
    core_radius_1=1.0,
    mass_to_light_ratio=1.0,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("EllipticalChameleon calculation_time = {}".format(calculation_time))

mass_profile = al.mp.SphericalChameleon(
    centre=(0.0, 0.0),
    intensity=1.0,
    core_radius_0=0.05,
    core_radius_1=1.0,
    mass_to_light_ratio=1.0,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("SphericalChameleon calculation_time = {}".format(calculation_time))

mass_profile = al.mp.EllipticalExponential(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=1.0,
    mass_to_light_ratio=1.0,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("EllipticalExponential calculation_time = {}".format(calculation_time))

mass_profile = al.mp.SphericalExponential(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, mass_to_light_ratio=1.0
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("SphericalExponential calculation_time = {}".format(calculation_time))

mass_profile = al.mp.EllipticalDevVaucouleurs(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=1.0,
    mass_to_light_ratio=1.0,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("EllipticalDevVaucouleurs calculation_time = {}".format(calculation_time))

mass_profile = al.mp.SphericalDevVaucouleurs(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, mass_to_light_ratio=1.0
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("SphericalDevVaucouleurs calculation_time = {}".format(calculation_time))

mass_profile = al.mp.EllipticalSersic(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
    mass_to_light_ratio=1.0,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("EllipticalSersic calculation_time = {}".format(calculation_time))

mass_profile = al.mp.SphericalSersic(
    centre=(0.0, 0.0),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
    mass_to_light_ratio=1.0,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("SphericalSersic calculation_time = {}".format(calculation_time))

mass_profile = al.mp.EllipticalSersicRadialGradient(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
    mass_to_light_ratio=1.0,
    mass_to_light_gradient=-1.0,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print(
    "EllipticalSersicRadialGradient (gradient = -1.0) calculation_time = {}".format(
        calculation_time
    )
)

mass_profile = al.mp.SphericalSersicRadialGradient(
    centre=(0.0, 0.0),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
    mass_to_light_ratio=1.0,
    mass_to_light_gradient=-1.0,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print(
    "SphericalSersicRadialGradient (gradient = -1.0) calculation_time = {}".format(
        calculation_time
    )
)

mass_profile = al.mp.EllipticalSersicRadialGradient(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
    mass_to_light_ratio=1.0,
    mass_to_light_gradient=1.0,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print(
    "EllipticalSersicRadialGradient (gradient = 1.0) calculation_time = {}".format(
        calculation_time
    )
)

mass_profile = al.mp.SphericalSersicRadialGradient(
    centre=(0.0, 0.0),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
    mass_to_light_ratio=1.0,
    mass_to_light_gradient=1.0,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print(
    "SphericalSersicRadialGradient (gradient = 1.0) calculation_time = {}".format(
        calculation_time
    )
)

mass_profile = al.mp.EllipticalCoreSersic(
    centre=(1.0, 2.0),
    elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.5, phi=70.0),
    intensity_break=0.45,
    effective_radius=0.5,
    radius_break=0.01,
    gamma=0.0,
    alpha=2.0,
    sersic_index=2.2,
)

calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("ElliptcialCoreSersic calculation_time = {}".format(calculation_time))

mass_profile = al.mp.SphericalCoreSersic(
    centre=(1.0, 2.0),
    intensity_break=0.45,
    effective_radius=0.5,
    radius_break=0.01,
    gamma=0.0,
    alpha=2.0,
    sersic_index=2.2,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("SphericalCoreSersic calculation_time = {}".format(calculation_time))

mass_profile = al.mp.EllipticalGaussian(
    centre=(0.0, 0.0),
    elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.6, phi=0.0),
    intensity=1.0,
    sigma=10.0,
    mass_to_light_ratio=1.0,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("ElliptcialGaussian calculation_time = {}".format(calculation_time))
