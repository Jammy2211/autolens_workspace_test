"""
__PROFILING: DEFLECTIONS__ 

This profiling script times how long deflection angles take to compute for different `MassProfile`'s
in **PyAutoLens**.

Deflections angles calculations are performed following one of three methods:

 1) When analytic formulae for the deflection angles are available these are used via NumPy array calculations (e.g.,
 `EllIsothermal`).

 2) When not available, numerical integration may be performed via `pyquad`, a Python wrapper to the GSL integration
 libraries (e.g. `EllPowerLawCored).

 3) The `MassProfile` convergence may be decomposed into a superposition of 20-30 Gaussian's where analytic expressions
 of a Gaussians deflection angle then offer fast computation (see https://arxiv.org/abs/1906.08263).
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import autolens as al
import json
import time

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
repeats = 10

"""
The function below times the deflection angle calculation on an input `MassProfile`.
"""


def time_deflections_2d_from_grid(mass_profile):
    start = time.time()
    for i in range(repeats):
        mass_profile.deflections_2d_from_grid(grid=grid)
    return (time.time() - start) / repeats


"""
The profiling dictionary stores the run time of every stellar mass profile.
"""
profiling_dict = {}

"""
We now iterate through every dark mass profile in PyAutoLens and compute how long the deflection angle calculation
takes.
"""
mass_profile = al.mp.EllChameleon(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    core_radius_0=0.05,
    core_radius_1=1.0,
    mass_to_light_ratio=1.0,
)
profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)

mass_profile = al.mp.SphChameleon(
    centre=(0.0, 0.0),
    intensity=1.0,
    core_radius_0=0.05,
    core_radius_1=1.0,
    mass_to_light_ratio=1.0,
)
profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)

mass_profile = al.mp.EllExponential(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=1.0,
    mass_to_light_ratio=1.0,
)
profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)

mass_profile = al.mp.SphExponential(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, mass_to_light_ratio=1.0
)
profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)

mass_profile = al.mp.EllDevVaucouleurs(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=1.0,
    mass_to_light_ratio=1.0,
)
profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)

mass_profile = al.mp.SphDevVaucouleurs(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, mass_to_light_ratio=1.0
)
profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)

mass_profile = al.mp.EllSersic(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
    mass_to_light_ratio=1.0,
)
profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)

mass_profile = al.mp.SphSersic(
    centre=(0.0, 0.0),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
    mass_to_light_ratio=1.0,
)
profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)

mass_profile = al.mp.EllSersicRadialGradient(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
    mass_to_light_ratio=1.0,
    mass_to_light_gradient=-1.0,
)
profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)

mass_profile = al.mp.SphSersicRadialGradient(
    centre=(0.0, 0.0),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
    mass_to_light_ratio=1.0,
    mass_to_light_gradient=-1.0,
)
profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)

mass_profile = al.mp.EllSersicCore(
    centre=(1.0, 2.0),
    elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.5, angle=70.0),
    intensity_break=0.45,
    effective_radius=0.5,
    radius_break=0.01,
    gamma=0.0,
    alpha=2.0,
    sersic_index=2.2,
)

profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)

mass_profile = al.mp.SphSersicCore(
    centre=(1.0, 2.0),
    intensity_break=0.45,
    effective_radius=0.5,
    radius_break=0.01,
    gamma=0.0,
    alpha=2.0,
    sersic_index=2.2,
)
profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)

mass_profile = al.mp.EllGaussian(
    centre=(0.0, 0.0),
    elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.6, angle=0.0),
    intensity=1.0,
    sigma=10.0,
    mass_to_light_ratio=1.0,
)
profiling_dict[mass_profile.__class__.__name__] = time_deflections_2d_from_grid(
    mass_profile=mass_profile
)


"""
Print the profiling results of every `MassProfile` for command line output when running profiling scripts.
"""
for key, value in profiling_dict.items():
    print(key, value)

"""
Output the profiling run times as a dictionary so they can be used in `profiling/graphs.py` to create graphs of the
profile run times.

This is stored in a folder using the **PyAutoLens** version number so that profiling run times can be tracked through
**PyAutoLens** development.
"""
file_path = os.path.join("profiling", "times", al.__version__, "deflections")

if not os.path.exists(file_path):
    os.makedirs(file_path)

file_path = os.path.join(file_path, f"stellar_mass_profiles__pixels_{pixels}.json")

if os.path.exists(file_path):
    os.remove(file_path)

with open(file_path, "w") as outfile:
    json.dump(profiling_dict, outfile)
