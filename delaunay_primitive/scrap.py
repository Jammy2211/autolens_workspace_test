import numpy as np
import scipy.spatial
import jax
import jax.numpy as jnp


def scipy_delaunay_padded(points_np, max_simplices):
    tri = scipy.spatial.Delaunay(points_np)

    pts = tri.points  # same dtype as input
    simplices = tri.simplices.astype(np.int32)

    # Pad simplices to fixed size (max_simplices, 3)
    padded = -np.ones((max_simplices, 3), dtype=np.int32)
    padded[: simplices.shape[0]] = simplices

    return pts, padded



def jax_delaunay(points):
    N = points.shape[0]
    max_simplices = 2 * N

    pts_shape = jax.ShapeDtypeStruct((N, 2), points.dtype)
    simp_shape = jax.ShapeDtypeStruct((max_simplices, 3), jnp.int32)

    return jax.pure_callback(
        lambda pts: scipy_delaunay_padded(pts, max_simplices),
        (pts_shape, simp_shape),
        points,
    )




@jax.jit
def likelihood(x):
    pts, simplices = jax_delaunay(x)
    return jnp.sum(pts) + jnp.sum(simplices)

arr = np.load("delaunay.npy")
print(likelihood(arr))



# import numpy as np
# import scipy.spatial
# import jax
# import jax.numpy as jnp
#
# arr = np.load("delaunay.npy")
#
# @jax.jit
# def run_delaunay(arr):
#     # This line will fail under JIT tracing
#     return scipy.spatial.Delaunay(
#         np.asarray([arr[:, 0], arr[:, 1]]).T
#     )
#
# # Trigger the JIT trace and error
# run_delaunay(arr)
#


"""
Hi Everyone,

I am now thinking about how we inspect modeling results as a SWG, instead of it just being me who looks and makes those judgements.

My current system is that I produce a single image to inspect the result of each lens:

There are two formatting issues with these images I will fix sooner or later:

1) The RGB should "zoom in " onto the scale of the VIS images, I will fix this in the next iteration.
2) Weird misalignment issues between cut-outs, will probably take longer to fix as ugly as it is!

I currently then write down notes on each lens, normally breaking down into three judgements:

1) This is a good lens model to a genuine lens.
2) This is probably not a lens.
3) This is a lens but the lens model failed for [INSERT REASON HERE].

I basically want to start the process of democratizing this process, to improve the judgement made to each lens, have everyone more involved with the modeling and because its fun.

I have attached images like that above for model to ~320 Q1 lenses here:


"""