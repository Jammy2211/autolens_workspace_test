import jax
import jax.numpy as jnp

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


i = 13
source_grid_size = 5
shape_native = (source_grid_size, source_grid_size)

"""
__Source Plane Data Grid__
"""
import numpy as np

source_plane_data_grid = np.load("source_plane_data_grid.npy")

# source_plane_data_grid = aa.Grid2D.uniform(
#     shape_native=(source_grid_size, source_grid_size),
#     pixel_scales=0.4
# )

"""
__Unit Conversion and Transform__
"""
y, x = source_plane_data_grid[:, 0], source_plane_data_grid[:, 1]

ymin, ymax = y.min(), y.max()  # -1, +1
xmin, xmax = x.min(), x.max()  # -1, +1

y_unit = (y - ymin) / (ymax - ymin)  # maps -1→0, +1→1
x_unit = (x - xmin) / (xmax - xmin)  # maps -

source_plane_data_grid_unit = jnp.stack([y_unit, x_unit], axis=1)


source_plane_data_grid_over_sampled_transformed = (
    adaptive_rectangular_transformed_grid_from(
        source_plane_data_grid, source_plane_data_grid_unit
    )
)

"""
__Source Plane Mesh Grid__
"""
source_plane_mesh_grid = aa.Grid2D.uniform(
    shape_native=(source_grid_size, source_grid_size), pixel_scales=0.5
)

"""
__Unit Conversion and Transform__
"""
y, x = source_plane_mesh_grid[:, 0], source_plane_mesh_grid[:, 1]

ymin, ymax = y.min(), y.max()  # -1, +1
xmin, xmax = x.min(), x.max()  # -1, +1

y_unit = (y - ymin) / (ymax - ymin)  # maps -1→0, +1→1
x_unit = (x - xmin) / (xmax - xmin)  # maps -1→0, +1→1

source_plane_mesh_grid_unit = jnp.stack([y_unit, x_unit], axis=1)

source_plane_mesh_grid_transformed = adaptive_rectangular_transformed_grid_from(
    source_plane_data_grid, source_plane_mesh_grid_unit
)

"""
__Indicies calculation copy and pasted from mapper_util__
"""

# --- Step 1. Normalize grid ---
mu = source_plane_data_grid.mean(axis=0)
scale = source_plane_data_grid.std(axis=0).min()
source_grid_scaled = (source_plane_data_grid - mu) / scale

# --- Step 2. Build transforms ---
transform, inv_transform = create_transforms(source_grid_scaled)

# --- Step 3. Transform oversampled grid into index space ---
grid_over_sampled_scaled = (source_plane_data_grid - mu) / scale
grid_over_sampled_transformed = transform(grid_over_sampled_scaled)
grid_over_index = source_grid_size * grid_over_sampled_transformed

# --- Step 4. Floor/ceil indices ---
ix_down = jnp.floor(grid_over_index[:, 0])
ix_up = jnp.ceil(grid_over_index[:, 0])
iy_down = jnp.floor(grid_over_index[:, 1])
iy_up = jnp.ceil(grid_over_index[:, 1])

# --- Step 5. Four corners ---
idx_tl = jnp.stack([ix_up, iy_down], axis=1)
idx_tr = jnp.stack([ix_up, iy_up], axis=1)
idx_br = jnp.stack([ix_down, iy_up], axis=1)
idx_bl = jnp.stack([ix_down, iy_down], axis=1)


# --- Step 6. Flatten indices ---
def flatten(idx, n):
    row = n - idx[:, 0]
    col = idx[:, 1]
    return row * n + col


flat_tl = flatten(idx_tl, source_grid_size)
flat_tr = flatten(idx_tr, source_grid_size)
flat_bl = flatten(idx_bl, source_grid_size)
flat_br = flatten(idx_br, source_grid_size)

# --- Step 8. Stack outputs ---
flat_indices = jnp.stack([flat_tl, flat_tr, flat_bl, flat_br], axis=1).astype("int64")

# --- Step 7. Bilinear interpolation weights ---
t_row = (grid_over_index[:, 0] - ix_down) / (ix_up - ix_down + 1e-12)
t_col = (grid_over_index[:, 1] - iy_down) / (iy_up - iy_down + 1e-12)

# Weights
w_tl = (1 - t_row) * (1 - t_col)
w_tr = (1 - t_row) * t_col
w_bl = t_row * (1 - t_col)
w_br = t_row * t_col
weights = jnp.stack([w_tl, w_tr, w_bl, w_br], axis=1)


# print("Source Plane Mesh Grid (Arc second space)")
# print(source_plane_mesh_grid)
#
# print("Source Plane Mesh Grid Transformed (Arc second space)")
# print(source_plane_mesh_grid_transformed)
#
# # print("Grid Transformed (Arc second space)")
# # print(source_plane_data_grid_over_sampled_transformed)
#
# # print("(y,x) coordinate to pair (unit space)")
# # print(grid_over_sampled_transformed[i])
#
# print("(y,x) coordinate to pair (arc seconds space)")
# # print(source_plane_data_grid[i])
# print(source_plane_data_grid_over_sampled_transformed[i])
#
# # print("Radial distance from transformed point to mesh grid center")
# # print(jnp.sqrt(jnp.sum((source_plane_mesh_grid_transformed - source_plane_data_grid_over_sampled_transformed[i])**2, axis=1)))
#
# print("Indexes of the 4 nearest pixels")
# print(flat_tl[i], flat_tr[i], flat_bl[i], flat_br[i])
#
# print("Their paired source plane mesh coordinates (arc seconds space)")
# print(source_plane_mesh_grid_transformed[flat_indices[i, 0]], source_plane_mesh_grid_transformed[flat_indices[i, 1]],
#       source_plane_mesh_grid_transformed[flat_indices[i, 2]], source_plane_mesh_grid_transformed[flat_indices[i, 3]])
#
# print("Weights of the 4 nearest pixels")
# print(w_tl[i], w_tr[i], w_bl[i], w_br[i])
#
# mappings, weights = (
#     aa.util.mapper.adaptive_rectangular_mappings_weights_via_interpolation_from(
#         source_grid_size=shape_native[0],
#         source_plane_data_grid=source_plane_data_grid,
#         source_plane_data_grid_over_sampled=jnp.array(
#             source_plane_data_grid
#         ),
#     )
# )
#
#
# print("Mappings of the 4 nearest pixels")
# print(mappings[i,0], mappings[i,1], mappings[i,2], mappings[i,3])
# print("Weights of the 4 nearest pixels")
# print(weights[i,0], weights[i,1], weights[i,2], weights[i,3])
#
#
#
#
#
# print(pixel_areas)
#
#
# edges = jnp.linspace(0, 1, shape_native[0] + 1)
# edges_reshaped = jnp.stack([edges, edges]).T
#
# edges_transformed = adaptive_rectangular_transformed_grid_from(
#     source_plane_data_grid,
#     edges_reshaped
# )
#
# edges_transformed_dense = np.moveaxis(
#     np.stack(np.meshgrid(*edges_transformed.T)),
#     0,
#     2
# )
#
# print(edges_transformed_dense.shape)
#
# print(edges_transformed_dense[0, 0, 0])
# print(edges_transformed_dense[0, 0, 1])
