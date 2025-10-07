import numpy as np
import jax.numpy as jnp


import numpy as np
import jax
import jax.numpy as jnp


def make_sparse_mapping_coo(
    image_pixels=10000, source_pixels=1000, nnz_per_col=4, seed=0
):
    """
    Create mapping matrix [image_pixels, source_pixels] in COO format:
    row indices, col indices, values.
    """
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []

    for col in range(source_pixels):
        r = rng.choice(image_pixels, size=nnz_per_col, replace=False)
        v = rng.random(nnz_per_col)
        rows.append(r)
        cols.append(np.full(nnz_per_col, col))
        vals.append(v)

    return (
        np.concatenate(rows),
        np.concatenate(cols),
        np.concatenate(vals),
    )


def make_sparse_wtilde_coo(image_pixels=10000, nnz_per_row=1000, seed=1):
    """
    Create symmetric w_tilde [image_pixels, image_pixels] in COO format.
    """
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []

    for i in range(image_pixels):
        c = rng.choice(image_pixels, size=nnz_per_row, replace=False)
        v = rng.random(nnz_per_row)

        rows.append(np.full(nnz_per_row, i))
        cols.append(c)
        vals.append(v)

        # symmetry: add (c, i) too
        rows.append(c)
        cols.append(np.full(nnz_per_row, i))
        vals.append(v)

    # diagonal
    diag = np.arange(image_pixels)
    diag_vals = rng.random(image_pixels) + nnz_per_row
    rows.append(diag)
    cols.append(diag)
    vals.append(diag_vals)

    return (
        np.concatenate(rows),
        np.concatenate(cols),
        np.concatenate(vals),
    )


def scatter_to_dense(rows, cols, vals, shape):
    """
    Convert COO triples into a dense JAX array using scatter_add.
    """
    arr = jnp.zeros(shape, dtype=jnp.float32)
    arr = arr.at[(rows, cols)].add(vals)
    return arr


# Example usage
M_rows, M_cols, M_vals = make_sparse_mapping_coo()
W_rows, W_cols, W_vals = make_sparse_wtilde_coo()

# Move to JAX
M_rows, M_cols, M_vals = map(jnp.array, (M_rows, M_cols, M_vals))
W_rows, W_cols, W_vals = map(jnp.array, (W_rows, W_cols, W_vals))

M = scatter_to_dense(M_rows, M_cols, M_vals, (10000, 1000))
W = scatter_to_dense(W_rows, W_cols, W_vals, (10000, 10000))

print("M shape:", M.shape, "nnz approx:", M_vals.shape[0])
print("W shape:", W.shape, "nnz approx:", W_vals.shape[0] // 2, "unique pairs")


mapping_matrix = jnp.array(M)
w_tilde = jnp.array(W)

import jax
import time


def curvature_matrix_via_w_tilde_from(
    w_tilde: np.ndarray, mapping_matrix: np.ndarray
) -> np.ndarray:
    return jnp.dot(mapping_matrix.T, jnp.dot(w_tilde, mapping_matrix))


jitted_curvature_matrix_via_w_tilde_from = jax.jit(curvature_matrix_via_w_tilde_from)

start_time = time.time()

curvature_matrix = jitted_curvature_matrix_via_w_tilde_from(
    w_tilde=w_tilde, mapping_matrix=mapping_matrix
)
print(curvature_matrix.shape, curvature_matrix.sum())

end_time = time.time()
print(f"Time JAX jit curvature_matrix w_tilde: {end_time - start_time}")


import numpy as np
import jax
import jax.numpy as jnp

import numpy as np


def make_sparse_matrices(
    image_pixels=10000, source_pixels=1000, m_nnz_per_col=4, w_nnz_per_row=1000, seed=0
):
    rng = np.random.default_rng(seed)

    # Mapping matrix M [image_pixels, source_pixels]
    M_rows, M_cols, M_vals = [], [], []
    for col in range(source_pixels):
        rows = rng.choice(image_pixels, size=m_nnz_per_col, replace=False)
        vals = rng.random(m_nnz_per_col)
        M_rows.extend(rows)
        M_cols.extend([col] * m_nnz_per_col)
        M_vals.extend(vals)
    M_rows, M_cols, M_vals = map(np.array, (M_rows, M_cols, M_vals))

    # W [image_pixels, image_pixels] symmetric
    W_rows, W_cols, W_vals = [], [], []
    for i in range(image_pixels):
        js = rng.choice(image_pixels, size=w_nnz_per_row, replace=False)
        vals = rng.random(w_nnz_per_row)
        for j, v in zip(js, vals):
            W_rows.append(i)
            W_cols.append(j)
            W_vals.append(v)
            if i != j:  # enforce symmetry
                W_rows.append(j)
                W_cols.append(i)
                W_vals.append(v)
    W_rows, W_cols, W_vals = map(np.array, (W_rows, W_cols, W_vals))

    return (M_rows, M_cols, M_vals), (W_rows, W_cols, W_vals)


from collections import defaultdict


def build_adjacency_M(M_rows, M_cols, M_vals):
    M_adj = defaultdict(list)
    for r, c, v in zip(M_rows, M_cols, M_vals):
        M_adj[r].append((c, v))

    return M_adj


def build_adjacency_W(W_rows, W_cols, W_vals):

    W_adj = defaultdict(list)
    for r, c, v in zip(W_rows, W_cols, W_vals):
        W_adj[r].append((c, v))

    return W_adj


import jax.numpy as jnp


def expand_to_scatter_jax(M_idx, M_val, W_i, W_j, W_val):
    """
    M_idx: [n_image, max_deg]  (indices into source plane)
    M_val: [n_image, max_deg]  (weights)
    W_i, W_j, W_val: [n_pairs] precomputed
    """

    def process_pair(i, j, w):
        a = M_idx[i]  # [max_deg]
        va = M_val[i]
        b = M_idx[j]  # [max_deg]
        vb = M_val[j]

        # Outer product contributions (broadcasting)
        flat_a = jnp.repeat(a, len(b))
        flat_b = jnp.tile(b, len(a))
        flat_val = jnp.repeat(va, len(b)) * w * jnp.tile(vb, len(a))

        return flat_a, flat_b, flat_val

    flat_a, flat_b, flat_val = jax.vmap(process_pair)(W_i, W_j, W_val)

    return (
        flat_a.reshape(-1),
        flat_b.reshape(-1),
        flat_val.reshape(-1),
    )


def scatter_F(flat_a, flat_b, flat_val, source_pixels):
    idx = (flat_a, flat_b)
    F = jnp.zeros((source_pixels, source_pixels))
    F = F.at[idx].add(flat_val)
    return F


import time

# 1. Generate sparse test data
# (M_rows, M_cols, M_vals), (W_rows, W_cols, W_vals) = make_sparse_matrices()

# 2. Build adjacency
M_rows, M_cols, M_vals = map(np.array, (M_rows, M_cols, M_vals))

# print(M_rows)
# print(M_cols)
# print(M_vals)
# ffff

W_rows, W_cols, W_vals = map(np.array, (W_rows, W_cols, W_vals))

import time

W_adj = build_adjacency_W(W_rows, W_cols, W_vals)


def precompute_W_pairs(W_adj):
    i_list, j_list, wval_list = [], [], []
    for i, neighbors in W_adj.items():
        for j, wval in neighbors:
            i_list.append(i)
            j_list.append(j)
            wval_list.append(wval)
    return (
        jnp.array(i_list, dtype=jnp.int32),
        jnp.array(j_list, dtype=jnp.int32),
        jnp.array(wval_list, dtype=jnp.float32),
    )


# Example:
W_i, W_j, W_val = precompute_W_pairs(W_adj)

start_time = time.time()

M_adj = build_adjacency_M(M_rows, M_cols, M_vals)

print(f"Time build adjacency M: {time.time() - start_time}")

# 3. Expand contributions

import time

start_time = time.time()

flat_a, flat_b, flat_val = expand_to_scatter_jax(M_idx, M_val, W_i, W_j, W_val)

print(f"Time expand to scatter: {time.time() - start_time}")

# 4. Scatter in JAX

start_time = time.time()

scatter_F_jit = jax.jit(scatter_F, static_argnums=(3,))

F = scatter_F_jit(flat_a, flat_b, flat_val, source_pixels=1000)

end_time = time.time()

print(f"Time JAX scatter curvature_matrix w_tilde: {end_time - start_time}")

print(F.shape, F.sum())
