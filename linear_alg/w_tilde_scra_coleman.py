import numpy as np
import jax.numpy as jnp


import numpy as np
import jax
import jax.numpy as jnp


def make_sparse_mapping_coo(
    image_pixels=10000, source_pixels=1000, nnz_per_col=10, seed=0
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


def make_sparse_wtilde_coo(image_pixels=10000, nnz_per_row=2000, seed=1):
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


def build_padded_adj_W(W_dict, n_rows, max_deg=None):
    """
    Convert a dict-of-lists adjacency W into padded arrays.

    Parameters
    ----------
    W_dict : dict
        Dictionary mapping row -> list of (col, val).
    n_rows : int
        Total number of rows.
    max_deg : int, optional
        Maximum degree (neighbors per row). If None, inferred from data.

    Returns
    -------
    W_cols : jnp.ndarray, shape (n_rows, max_deg)
        Padded column indices, -1 for unused slots.
    W_vals : jnp.ndarray, shape (n_rows, max_deg)
        Padded values, 0.0 for unused slots.
    """
    if max_deg is None:
        max_deg = max(len(neigh) for neigh in W_dict.values())

    cols = np.full((n_rows, max_deg), -1, dtype=np.int32)
    vals = np.zeros((n_rows, max_deg), dtype=np.float32)

    for r, neigh in W_dict.items():
        for j, (c, v) in enumerate(neigh):
            cols[r, j] = c
            vals[r, j] = v

    return jnp.array(cols), jnp.array(vals)


def build_padded_adj_M(M_rows, M_cols, M_vals, n_rows, max_deg):
    # Allocate padded arrays
    cols = jnp.full((n_rows, max_deg), -1)
    vals = jnp.zeros((n_rows, max_deg))

    # Scatter into them
    cols = cols.at[M_rows, jnp.arange(len(M_rows)) % max_deg].set(M_cols)
    vals = vals.at[M_rows, jnp.arange(len(M_rows)) % max_deg].set(M_vals)

    return cols, vals


def build_adjacency_W(W_rows, W_cols, W_vals):

    W_adj = defaultdict(list)
    for r, c, v in zip(W_rows, W_cols, W_vals):
        W_adj[r].append((c, v))

    return W_adj


import jax.numpy as jnp

import pickle


# def expand_to_scatter(M_adj, W_adj):
#     flat_a, flat_b, flat_val = [], [], []
#
#     for i, ma_list in M_adj.items():
#         for j, wval in W_adj[i]:
#             if j not in M_adj:  # skip if j has no M entries
#                 continue
#             for a, va in ma_list:
#                 for b, vb in M_adj[j]:
#                     flat_a.append(a)
#                     flat_b.append(b)
#                     flat_val.append(va * wval * vb)
#
#     return (np.array(flat_a), np.array(flat_b), np.array(flat_val))


def expand_to_scatter(M_cols, M_vals, W_cols, W_vals):
    """
    M_cols, M_vals: (n_rows, max_deg_M)
    W_cols, W_vals: (n_rows, max_deg_W)
    """
    n_rows, max_deg_M = M_cols.shape
    _, max_deg_W = W_cols.shape

    # For each row i: M[i,:] pairs with W[i,:]
    # Shape (n_rows, max_deg_W, max_deg_M)
    a = jnp.broadcast_to(M_cols[:, None, :], (n_rows, max_deg_W, max_deg_M))
    va = jnp.broadcast_to(M_vals[:, None, :], (n_rows, max_deg_W, max_deg_M))

    j = jnp.broadcast_to(W_cols[:, :, None], (n_rows, max_deg_W, max_deg_M))
    wval = jnp.broadcast_to(W_vals[:, :, None], (n_rows, max_deg_W, max_deg_M))

    # Gather M[j,:] for each neighbor j
    # j has shape (n_rows, max_deg_W, max_deg_M)
    b = jnp.take_along_axis(M_cols[None, :, :], j[..., None], axis=1).squeeze(-1)
    vb = jnp.take_along_axis(M_vals[None, :, :], j[..., None], axis=1).squeeze(-1)

    # Multiply values
    val = va * wval * vb

    # Flatten everything to 1D
    flat_a = a.ravel()
    flat_b = b.ravel()
    flat_val = val.ravel()

    return flat_a, flat_b, flat_val


def scatter_F(flat_a, flat_b, flat_val, source_pixels):
    idx = (flat_a, flat_b)
    F = jnp.zeros((source_pixels, source_pixels))
    F = F.at[idx].add(flat_val)
    return F


import time

# 1. Generate sparse test data
# (M_rows, M_cols, M_vals), (W_rows, W_cols, W_vals) = make_sparse_matrices()

# 2. Build adjacency
M_rows, M_cols, M_vals = map(jnp.array, (M_rows, M_cols, M_vals))
W_rows, W_cols, W_vals = map(np.array, (W_rows, W_cols, W_vals))

import time

W_adj = build_adjacency_W(W_rows, W_cols, W_vals)
W_cols, W_vals = build_padded_adj_W(W_adj, n_rows=10000)

build_padded_adj_M_jit = jax.jit(build_padded_adj_M, static_argnums=(3, 4))

start_time = time.time()

M_cols, M_vals = build_padded_adj_M_jit(
    M_rows, M_cols, M_vals, n_rows=10000, max_deg=7000
)


print(f"Time build adjacency M: {time.time() - start_time}")


# with open("M_adj.pkl", "wb") as f:
#     pickle.dump(M_adj, f)
#
# with open("W_adj.pkl", "wb") as f:
#     pickle.dump(W_adj, f)


# 3. Expand contributions

import time

start_time = time.time()

flat_a, flat_b, flat_val = expand_to_scatter(M_cols, M_vals, W_cols, W_vals)

print(flat_a.shape)
print(flat_b.shape)
print(flat_val.shape)

# print(flat_a[0:100])
# print(flat_b[0:100])
# print(flat_val[0:100])

print(f"Time expand to scatter: {time.time() - start_time}")

# 4. Scatter in JAX

start_time = time.time()

scatter_F_jit = jax.jit(scatter_F, static_argnums=(3,))

F = scatter_F_jit(flat_a, flat_b, flat_val, source_pixels=1000)

print(F[0])
end_time = time.time()

print(f"Time JAX scatter curvature_matrix w_tilde: {end_time - start_time}")

print(np.max(F - curvature_matrix))
