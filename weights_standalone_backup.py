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
n = source_grid_size

source_plane_mesh_grid = aa.Grid2D.uniform(
    shape_native=(source_grid_size, source_grid_size), pixel_scales=0.5
)

source_plane_data_grid = aa.Grid2D.uniform(
    shape_native=(source_grid_size, source_grid_size), pixel_scales=0.4
)
source_plane_data_grid_over_sampled = source_plane_data_grid

source_plane_data_grid_over_sampled_transformed = (
    adaptive_rectangular_transformed_grid_from(
        source_plane_data_grid.array, source_plane_data_grid_over_sampled.array
    )
)


# --- Step 1. Normalize grid ---
mu = source_plane_data_grid.mean(axis=0)
scale = source_plane_data_grid.std(axis=0).min()
source_grid_scaled = (source_plane_data_grid - mu) / scale

# --- Step 2. Build transforms ---
transform, inv_transform = create_transforms(source_grid_scaled.array)

# --- Step 3. Transform oversampled grid into index space ---
grid_over_sampled_scaled = (source_plane_data_grid_over_sampled - mu) / scale
grid_over_sampled_transformed = transform(grid_over_sampled_scaled.array)
grid_over_index = source_grid_size * grid_over_sampled_transformed

# --- Step 4. Floor/ceil indices ---
iy_down = jnp.floor(grid_over_index[:, 0]).astype(jnp.int32)  # row (y)
iy_up = jnp.ceil(grid_over_index[:, 0]).astype(jnp.int32)
ix_down = jnp.floor(grid_over_index[:, 1]).astype(jnp.int32)  # col (x)
ix_up = jnp.ceil(grid_over_index[:, 1]).astype(jnp.int32)

# clamp to grid interior
iy_down = jnp.clip(iy_down, 0, n - 2)
iy_up = jnp.clip(iy_up, 1, n - 1)
ix_down = jnp.clip(ix_down, 0, n - 2)
ix_up = jnp.clip(ix_up, 1, n - 1)

# --- Step 5. Four corners (row, col) ---

print(iy_up[i], iy_down[i], ix_up[i], ix_down[i])

row_up = (n - 1) - iy_up  # vertical flip
row_down = (n - 1) - iy_down
col_left = ix_down
col_right = ix_up

# Corners in (row, col) order
tl = jnp.stack([col_left, row_up], axis=1)
tr = jnp.stack([col_left, row_down], axis=1)
bl = jnp.stack([col_right, row_up], axis=1)
br = jnp.stack([col_right, row_down], axis=1)


# --- Step 6. Flatten indices ---
def flatten(idx):
    row = idx[:, 0]  # flip y so row=0 is top
    col = idx[:, 1]
    return (row * n + col).astype(jnp.int32)


print(tl[i], tr[i], bl[i], br[i])

flat_tl = flatten(tl)
flat_tr = flatten(tr)
flat_bl = flatten(bl)
flat_br = flatten(br)

flat_indices = jnp.stack([flat_tl, flat_tr, flat_bl, flat_br], axis=1).astype("int64")

print("Source Plane Mesh Grid (Arc second space)")
print(source_plane_mesh_grid)

print("Source Plane Mesh Grid Transformed (Arc second space)")
print(source_plane_mesh_grid_transformed)

# print("Grid Transformed (Arc second space)")
# print(source_plane_data_grid_over_sampled_transformed)

# print("(y,x) coordinate to pair (unit space)")
# print(grid_over_sampled_transformed[i])

print("(y,x) coordinate to pair (arc seconds space)")
print(source_plane_data_grid_over_sampled_transformed[i])

print("Indexes of the 4 nearest pixels")
print(flat_tl[i], flat_tr[i], flat_bl[i], flat_br[i])

print("Their paired source plane mesh coordinates (arc seconds space)")
print(
    source_plane_mesh_grid_transformed[flat_indices[i, 0]],
    source_plane_mesh_grid_transformed[flat_indices[i, 1]],
    source_plane_mesh_grid_transformed[flat_indices[i, 2]],
    source_plane_mesh_grid_transformed[flat_indices[i, 3]],
)


# --- Step 7. Bilinear interpolation weights ---

ix_down = jnp.floor(grid_over_index[:, 0])
ix_up = jnp.ceil(grid_over_index[:, 0])
iy_down = jnp.floor(grid_over_index[:, 1])
iy_up = jnp.ceil(grid_over_index[:, 1])

idx_down = jnp.stack([ix_down, iy_down], axis=1)
idx_up = jnp.stack([ix_up, iy_up], axis=1)

delta_up = idx_up - grid_over_index
delta_down = grid_over_index - idx_down

w_tl = delta_up[:, 0] * delta_up[:, 1]
w_bl = delta_up[:, 0] * delta_down[:, 1]
w_tr = delta_down[:, 0] * delta_up[:, 1]
w_br = delta_down[:, 0] * delta_down[:, 1]


# --- Step 8. Stack outputs ---

# weights = jnp.stack([w_tl, w_tr, w_bl, w_br], axis=1)


print("Weights of the 4 nearest pixels")
print(w_tl[i], w_bl[i], w_tr[i], w_br[i])
