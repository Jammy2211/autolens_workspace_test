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


def adaptive_rectangular_areas_from(source_grid_size, source_plane_data_grid):

    pixel_edges_1d = jnp.linspace(0, 1, source_grid_size + 1)

    mu = source_plane_data_grid.mean(axis=0)
    scale = source_plane_data_grid.std(axis=0).min()
    source_grid_scaled = (source_plane_data_grid - mu) / scale

    transform, inv_transform = create_transforms(source_grid_scaled)

    def inv_full(U):
        return inv_transform(U) * scale + mu

    pixel_edges = inv_full(jnp.stack([pixel_edges_1d, pixel_edges_1d]).T)
    pixel_lengths = jnp.diff(pixel_edges, axis=0).squeeze()  # shape (N_source, 2)

    dy = pixel_lengths[:, 0]
    dx = pixel_lengths[:, 1]

    return jnp.outer(dy, dx).flatten()


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


"""
__Indicies calculation copy and pasted from mapper_util__
"""
N = source_grid_size

# --- Step 1. Normalize grid ---
mu = source_plane_data_grid.mean(axis=0)
scale = source_plane_data_grid.std(axis=0).min()
source_grid_scaled = (source_plane_data_grid - mu) / scale

# --- Step 2. Build transforms ---
transform, inv_transform = create_transforms(source_grid_scaled)

# --- Step 3. Transform oversampled grid into unit space [0,1]^2 ---
grid_over_sampled_scaled = (source_plane_data_grid - mu) / scale
grid_over_sampled_transformed = transform(grid_over_sampled_scaled)  # (N,2)
grid_over_index = N * grid_over_sampled_transformed

# --- Step 5. Per-pixel dy, dx spans from inverse transform ---
pixel_edges_1d = jnp.linspace(0, 1, N + 1)


def inv_full(U):
    return inv_transform(U) * scale + mu


pixel_edges = inv_full(jnp.stack([pixel_edges_1d, pixel_edges_1d]).T)
pixel_lengths = jnp.diff(pixel_edges, axis=0).squeeze()  # (N,2)

dy_all = pixel_lengths[:, 0]
dx_all = pixel_lengths[:, 1]

# Flatten to 1D for lookup
dy_flat = jnp.repeat(dy_all, N) / (10 * scale)  # convert to unit space
dx_flat = jnp.tile(dx_all, N) / (10 * scale)

# --- Step 6. Find pixel index for each point ---
ix = jnp.clip(jnp.floor(grid_over_index[:, 1]).astype(int), 0, N - 1)
iy = jnp.clip(jnp.floor(grid_over_index[:, 0]).astype(int), 0, N - 1)
pixel_index = iy * N + ix

# Assign spans per point
dy_point = dy_flat[pixel_index]
dx_point = dx_flat[pixel_index]

# --- Step 7. Build anisotropic cross points in unit space ---
y0 = grid_over_sampled_transformed[:, 0]
x0 = grid_over_sampled_transformed[:, 1]

pt_up = jnp.stack([y0 + dy_point, x0], axis=1)
pt_down = jnp.stack([y0 - dy_point, x0], axis=1)
pt_right = jnp.stack([y0, x0 + dx_point], axis=1)
pt_left = jnp.stack([y0, x0 - dx_point], axis=1)

cross_pts_unit = jnp.stack(
    [pt_up, pt_right, pt_down, pt_left], axis=1
)  # (n_points,4,2)
cross_pts_index = N * cross_pts_unit

# --- Step 8. Bilinear weights at each cross point ---
y = cross_pts_index[..., 0]
x = cross_pts_index[..., 1]

iy0 = jnp.clip(jnp.floor(y).astype(int), 0, N - 2)
ix0 = jnp.clip(jnp.floor(x).astype(int), 0, N - 2)

ty = y - iy0
tx = x - ix0

w_bl = (1 - tx) * (1 - ty)
w_br = tx * (1 - ty)
w_tl = (1 - tx) * ty
w_tr = tx * ty

weights_all = jnp.stack([w_bl, w_br, w_tl, w_tr], axis=-1)  # (n_points,4cross,4neigh)

# --- Step 9. Average across cross points ---
weights = weights_all.mean(axis=1)  # (n_points,4)

print(weights[100])
print(weights[101])
print(weights[102])
print(weights[103])

# --- Step 10. Build neighbor indices (flat) ---
iy1 = iy0 + 1
ix1 = ix0 + 1

flat_bl = iy0 * N + ix0
flat_br = iy0 * N + ix1
flat_tl = iy1 * N + ix0
flat_tr = iy1 * N + ix1

flat_indices = jnp.stack([flat_bl, flat_br, flat_tl, flat_tr], axis=1)  # (n_points,4)
