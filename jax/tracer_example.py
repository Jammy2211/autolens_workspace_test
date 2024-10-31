"""
Tracer Simple
=============

This is a very short example to help us JAX-ify the Tracer.

Basically, the SIE and Power Law mass profiles should now support JAX, but the full lensing calculation goes
via the `Tracer`.

This example runs a JAX-ed `Tracer` calculation so we can sort it.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import jax
from jax import grad
from os import path

import autolens as al


"""
__Grid__
"""
mask = al.Mask2D.circular(shape_native=(100, 100), pixel_scales=0.05, radius=3.0)

grid = al.Grid2D.from_mask(
    mask=mask,
    over_sampling=al.OverSamplingUniform(sub_size=1)
)

"""
__SIE Mass Profile__

We first confirm JAX plays nice with the SIE.
"""
mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    ell_comps=(0.2, 0.2),
    einstein_radius=1.0,
)

grad = jax.jit(grad(mass.deflections_yx_2d_from))
grad(grid.array)

"""
__Tracer__

The Tracer takes a different path throuigh the source code, which is not fully JAX-ed yet.
"""
lens = al.Galaxy(redshift=0.5, mass=mass)
source = al.Galaxy(
    redshift=1.0,
    light=al.lp.Sersic(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0),
)

tracer = al.Tracer(galaxies=[lens, source])

grad = jax.jit(grad(tracer.deflections_yx_2d_from))
grad(grid.array)

"""
Checkout `autogalaxy_workspace/*/imaging/modeling/results.py` for a full description of the result object.
"""
