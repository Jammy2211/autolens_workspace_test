import numpy as np

import autoarray as aa

mask_size = 11
border_size = 2
kernel_size = 3

mask = np.full(shape=(mask_size, mask_size), fill_value=True)

mask[border_size : mask_size - border_size, border_size : mask_size - border_size] = (
    False
)

mask = aa.Mask2D(mask=mask, pixel_scales=1.0)

noise_map_native = np.ones(shape=(mask_size, mask_size))

noise_map_native = aa.Array2D(values=noise_map_native, mask=mask).native

kernel_native = aa.Kernel2D.ones(
    shape_native=(kernel_size, kernel_size), pixel_scales=1.0
).native

native_index_for_slim_index = np.array(
    noise_map_native.mask.derive_indexes.native_for_slim
)


w_tilde_curvature = aa.util.inversion_imaging_numba.w_tilde_curvature_imaging_from(
    noise_map_native=noise_map_native.array,
    kernel_native=kernel_native.array,
    native_index_for_slim_index=native_index_for_slim_index,
)

curvature_preload, curvature_indexes, curvature_lengths = (
    aa.util.inversion_imaging_numba.w_tilde_curvature_preload_imaging_from(
        noise_map_native=noise_map_native.array,
        kernel_native=kernel_native.array,
        native_index_for_slim_index=native_index_for_slim_index,
    )
)


from collections import defaultdict


def build_adjacency_W(W_rows, W_cols, W_vals):

    W_adj = defaultdict(list)
    for r, c, v in zip(W_rows, W_cols, W_vals):
        W_adj[r].append((c, v))

    return W_adj


import jax.numpy as jnp

# 1. Generate sparse test data
# (M_rows, M_cols, M_vals), (W_rows, W_cols, W_vals) = make_sparse_matrices()

# 2. Build adjacency
M_rows, M_cols, M_vals = map(np.array, (M_rows, M_cols, M_vals))
W_rows, W_cols, W_vals = map(np.array, (W_rows, W_cols, W_vals))

import time

start_time = time.time()


M_adj = build_adjacency_M(M_rows, M_cols, M_vals)

print(M_adj[0])
print(f"Time build adjacency M: {time.time() - start_time}")

W_adj = build_adjacency_W(W_rows, W_cols, W_vals)

# 3. Expand contributions

import time

start_time = time.time()

flat_a, flat_b, flat_val = expand_to_scatter(M_adj, W_adj)

print(flat_a.shape)
print(flat_b.shape)
print(flat_val.shape)

print(flat_a[0:100])
print(flat_b[0:100])
print(flat_val[0:100])

print(f"Time expand to scatter: {time.time() - start_time}")

# 4. Scatter in JAX

start_time = time.time()

scatter_F_jit = jax.jit(scatter_F, static_argnums=(3,))

F = scatter_F_jit(flat_a, flat_b, flat_val, source_pixels=1000)

print(F[0])
end_time = time.time()

print(f"Time JAX scatter curvature_matrix w_tilde: {end_time - start_time}")

print(np.max(F - curvature_matrix))
