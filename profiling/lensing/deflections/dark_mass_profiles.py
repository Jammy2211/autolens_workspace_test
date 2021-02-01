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


mass_profile = al.mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=0.1, scale_radius=10.0)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("SphericalNFW calculation_time = {}".format(calculation_time))

mass_profile = al.mp.SphericalTruncatedNFW(
    centre=(0.0, 0.0), kappa_s=0.1, scale_radius=10.0, truncation_radius=5.0
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("SphericalTruncatedNFW calculation_time = {}".format(calculation_time))

mass_profile = al.mp.SphericalGeneralizedNFW(
    centre=(0.0, 0.0), kappa_s=0.1, scale_radius=10.0, inner_slope=0.5
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print(
    "SphericalGeneralizedNFW (inner_slope = 1.0) calculation_time = {}".format(
        calculation_time
    )
)

mass_profile = al.mp.EllipticalNFW(
    centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), kappa_s=0.1, scale_radius=10.0
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("EllipticalNFW calculation_time = {}".format(calculation_time))


mass_profile = al.mp.EllipticalGeneralizedNFW(
    centre=(0.0, 0.0),
    elliptical_comps=(0.111111, 0.0),
    kappa_s=0.1,
    scale_radius=10.0,
    inner_slope=1.8,
)
calculation_time = time_deflections_from_grid(mass_profile=mass_profile)
print("EllipticalGeneralizedNFW calculation_time = {}".format(calculation_time))
