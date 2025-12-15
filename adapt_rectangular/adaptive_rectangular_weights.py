import copy

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

import autoarray as aa


def forward_interp(xp, yp, x):
    return jax.vmap(jnp.interp, in_axes=(1, 1, None, None, None))(x, xp, yp, 0, 1).T


def reverse_interp(xp, yp, x):
    return jax.vmap(jnp.interp, in_axes=(1, None, 1))(x, xp, yp).T


def create_transforms(traced_points):
    # make functions that takes a set of traced points
    # stored in a (N, 2) array and return functions that
    # take in (N, 2) arrays and transform the values into
    # the range (0, 1) and the inverse transform
    N = traced_points.shape[0]  # // 2
    t = jnp.arange(1, N + 1) / (N + 1)

    sort_points = jnp.sort(traced_points, axis=0)  # [::2]

    transform = partial(forward_interp, sort_points, t)
    inv_transform = partial(reverse_interp, t, sort_points)
    return transform, inv_transform


def adaptive_rectangular_transformed_grid_from(source_plane_data_grid, grid):
    mu = source_plane_data_grid.mean(axis=0)
    scale = source_plane_data_grid.std(axis=0).min()

    source_grid_scaled = (source_plane_data_grid - mu) / scale

    transform, inv_transform = create_transforms(source_grid_scaled)

    def inv_full(U):
        return inv_transform(U) * scale + mu

    return inv_full(grid)


# Shape of Rectnagular grid before transform.
shape_native = (10, 10)

# Contains ray traced points in source plane without super sampling, shape corresponded to observed image.
source_plane_mesh_grid = np.load("source_plane_mesh_grid.npy")
source_plane_data_grid = np.load("source_plane_data_grid.npy")

source_plane_data_grid_transformed = (
    aa.util.mapper.adaptive_rectangular_transformed_grid_from(
        source_plane_data_grid, source_plane_data_grid
    )
)

source_plane_mesh_grid = aa.util.mapper.adaptive_rectangular_transformed_grid_from(
    source_plane_data_grid, source_plane_mesh_grid
)

print(source_plane_mesh_grid)

aa.util.mapper.adaptive_rectangular_mappings_weights_via_interpolation_from(
    source_grid_size=10,
    source_plane_data_grid=jnp.array(source_plane_data_grid),
    source_plane_data_grid_over_sampled=jnp.array(source_plane_data_grid),
    source_plane_mesh_grid=source_plane_mesh_grid,
)

print(source_plane_data_grid_transformed[40])
fff

# Apply transform to edges:
edges_transformed = adaptive_rectangular_transformed_grid_from(
    source_plane_data_grid=source_plane_data_grid, grid=edges_reshaped
)

# To turn the edges into a dense mesh of grid points
edges_transformed_dense = np.moveaxis(np.stack(np.meshgrid(*edges_transformed.T)), 0, 2)


C = jnp.arange(1, shape_native[0] * shape_native[1] + 1).reshape(
    shape_native[0], shape_native[1]
)

# Now make plot:
plt.pcolormesh(
    edges_transformed_dense[..., 0], edges_transformed_dense[..., 1], C, shading="flat"
)
plt.plot(*source_plane_data_grid.T, ",", color="C3")
plt.plot(
    edges_transformed_dense[..., 0].flatten(),
    edges_transformed_dense[..., 1].flatten(),
    ".",
    color="C1",
)
plt.savefig("adaptive_rectangular_grid.png", dpi=300)

# Ny, Nx = shape_native
# # Collect x and y edges: each pixel contributes (x0,x1) and (y0,y1)
# y_edges = jnp.unique(mesh.edges_transformed[...,0])  # shape (Ny+1,)
# x_edges = jnp.unique(mesh.edges_transformed[...,1])  # shape (Nx+1,)
#
# print(y_edges)
# print(x_edges)
# ffff
# #
# # X, Y = jnp.meshgrid(x_edges, y_edges)
#
# grid_edges_2d = mesh.edges_transformed.reshape(-1, 2).reshape(shape_native[0]+1, shape_native[1]+1, 2)
#
# print(grid_edges_2d.shape)
#
# C = jnp.arange(1, Ny * Nx + 1).reshape(Ny, Nx)


# print(mesh.edges_transformed)

# mappings, weights = (
#     mapper_util.rectangular_mappings_weights_via_interpolation_from(
#         shape_native=shape_native,
#         source_plane_mesh_grid=source_plane_mesh_grid,
#         source_plane_data_grid=source_plane_data_grid,
#     )
# )
#
# print(source_plane_data_grid_over_sampled)

# source_plane_data_grid_over_sampled[36,0] = 0.03
# source_plane_data_grid_over_sampled[36,1] = 0.03
#
# mappings, weights = (
#     mapper_util.adaptive_rectangular_mappings_weights_via_interpolation_from(
#         source_grid_size=32,
#         source_plane_data_grid=jnp.array(source_plane_data_grid),
#         source_plane_data_grid_over_sampled=jnp.array(source_plane_data_grid_over_sampled),
#     )
# )
#
# for i in range(len(mappings)):
#     print(i, source_plane_data_grid_over_sampled[i,:], mappings[i], weights[i])
#     if i == 36:
#         fff
