from functools import partial
import numpy as np
import jax.numpy as jnp

traced_points = jnp.array(np.load("adapt_rect/traced_points.npy", allow_pickle=True))
mesh_weight_map = jnp.array(np.load("adapt_rect/mesh_weight_map.npy", allow_pickle=True))
# mesh_weight_map = None
source_plane_data_grid = jnp.array(np.load("adapt_rect/source_plane_data_grid.npy", allow_pickle=True))
source_plane_data_grid_over_sampled = jnp.array(np.load("adapt_rect/source_plane_data_grid_over_sampled.npy", allow_pickle=True))

def forward_interp(xp, yp, x):

    import jax
    import jax.numpy as jnp
    return jax.vmap(jnp.interp, in_axes=(1, 1, 1, None, None), out_axes=(1))(x, xp, yp, 0, 1)


def reverse_interp(xp, yp, x):
    import jax
    import jax.numpy as jnp
    return jax.vmap(jnp.interp, in_axes=(1, 1, 1), out_axes=(1))(x, xp, yp)

def create_transforms(traced_points, mesh_weight_map = None):

    N = traced_points.shape[0]  # // 2

    if mesh_weight_map is None:
        t = jnp.arange(1, N + 1) / (N + 1)
        t = jnp.stack([t, t], axis=1)
        sort_points = jnp.sort(traced_points, axis=0)  # [::2]
    else:
        sdx = jnp.argsort(traced_points, axis=0)
        sort_points = jnp.take_along_axis(traced_points, sdx, axis=0)
        t = jnp.stack([mesh_weight_map, mesh_weight_map], axis=1)
        t = jnp.take_along_axis(t, sdx, axis=0)
        t = jnp.cumsum(t, axis=0)

    transform = partial(forward_interp, sort_points, t)
    inv_transform = partial(reverse_interp, t, sort_points)

    import matplotlib.pyplot as plt

    plt.plot(sort_points, t)
    plt.savefig("adapt_rect/transform_debug.png")

    return transform, inv_transform

"""
No mesh weight map case
"""
# --- Step 1. Normalize grid ---
mu = source_plane_data_grid.mean(axis=0)
scale = source_plane_data_grid.std(axis=0).min()
source_grid_scaled = (source_plane_data_grid - mu) / scale

# --- Step 2. Build transforms ---
transform, inv_transform = create_transforms(source_grid_scaled, mesh_weight_map=mesh_weight_map)

# --- Step 3. Transform oversampled grid into index space ---
grid_over_sampled_scaled = (source_plane_data_grid_over_sampled - mu) / scale
grid_over_sampled_transformed = transform(grid_over_sampled_scaled)

