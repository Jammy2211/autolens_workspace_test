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

import os
os.environ["USE_JAX"] = "1"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import autolens as al


"""
__Grid__
"""
mask = al.Mask2D.circular(
    shape_native=(100, 100),
    pixel_scales=0.05,
    radius=3.0
)

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


def deflection_scalar(y, x, pixel_scales, deflection_fn):
    # make a version of the deflection angle function
    # that takes in two floats and returns a two element vector 
    g = al.Grid2D.from_yx_1d(
        y=y.reshape(1),
        x=x.reshape(1),
        shape_native=(1, 1),
        pixel_scales=pixel_scales
    )
    return deflection_fn(g).squeeze()


def A_stack(y, x, pixel_scales, deflection_fn):
    # Area distortion matrix of the lens mapping.
    #
    # Take the jacobian of deflection_scalar and return a 2x2 array
    return jnp.stack(
        jax.jacfwd(
            deflection_scalar,
            argnums=(0, 1)
        )(y, x, pixel_scales, deflection_fn)
    )


def A(y, x, pixel_scales, deflection_fn):
    # Area distortion matrix of the lens mapping.
    #
    # Vectorize A_stack over an array of y and x values
    # (any input shape is accepted)
    return jnp.vectorize(
        jax.tree_util.Partial(
            A_stack,
            pixel_scales=pixel_scales,
            deflection_fn=deflection_fn
        ),
        signature='(),()->(i,i)'
    )(y, x)


@jax.jit
def A_grid(grid, deflection_fn):
    # Area distortion matrix of the lens mapping.
    #
    # Wrapper to A that takes in a grid object
    y = grid.array[:, 0]
    x = grid.array[:, 1]
    # The output will need to be wrapped in a vector2D class I think
    return A(y, x, grid.pixel_scales, deflection_fn)


# The deflection function needs to be wrapped in `jax.tree_util.Partial`
# to be a valid JAX input data type
A_grid(
    grid,
    jax.tree_util.Partial(mass.deflections_yx_2d_from)
)

"""
__Tracer__

The Tracer takes a different path throuigh the source code, which is not fully JAX-ed yet.
"""
lens = al.Galaxy(redshift=0.5, mass=mass)
source = al.Galaxy(redshift=1.0, light=al.lp.Sersic(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0))

tracer = al.Tracer(galaxies=[lens, source])

# check it works for a tracer object too
A_grid(
    grid,
    jax.tree_util.Partial(tracer.deflections_yx_2d_from)
)


"""
Checkout `autogalaxy_workspace/*/imaging/modeling/results.py` for a full description of the result object.
"""
