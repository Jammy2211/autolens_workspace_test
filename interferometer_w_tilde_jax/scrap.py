import numpy as np

# -------------------------
# Load inputs
# -------------------------
w_tilde_preload = np.load("w_tilde_preload.npy")  # (2y, 2x)
pix_indexes_for_sub_slim_index = np.load("pix_indexes_for_sub_slim_index.npy")  # (M, Pmax) int
pix_sizes_for_sub_slim_index = np.load("pix_sizes_for_sub_slim_index.npy")      # (M,) int
pix_weights_for_sub_slim_index = np.load("pix_weights_for_sub_slim_index.npy")  # (M, Pmax) float
native_index_for_slim_index = np.load("native_index_for_slim_index.npy")        # (M,) or (M,2)
total_mapper_pixels = int(np.load("pix_pixels.npy"))                             # S
curvature_matrix_true = np.load("curvature_matrix.npy")                          # (S,S)

# -------------------------
# Helpers: build COO for mapping F (image->source)
# -------------------------
def build_mapping_coo_np(pix_idx, pix_wts, pix_sizes):
    """
    Build COO arrays (rows, cols, vals) for sparse mapping matrix F of shape (M, S).
    Each image pixel m contributes to pix_sizes[m] source pixels with given weights.
    """
    M, Pmax = pix_idx.shape
    # mask valid entries
    mask = (np.arange(Pmax)[None, :] < pix_sizes[:, None])
    rows = np.repeat(np.arange(M), Pmax)[mask.ravel()]
    cols = pix_idx[mask].astype(np.int32)
    vals = pix_wts[mask].astype(np.float64)
    return rows.astype(np.int32), cols, vals

rows_np, cols_np, vals_np = build_mapping_coo_np(
    pix_indexes_for_sub_slim_index,
    pix_weights_for_sub_slim_index,
    pix_sizes_for_sub_slim_index,
)

# -------------------------
# Determine rectangular image shape from preload
# -------------------------
H2, W2 = w_tilde_preload.shape
assert H2 % 2 == 0 and W2 % 2 == 0
y_shape = H2 // 2
x_shape = W2 // 2
M_expected = y_shape * x_shape

# If your image pixels M aren't equal to y*x, you need a different embedding.
M = int(pix_sizes_for_sub_slim_index.shape[0])
assert M == M_expected, (
    f"Expected M=y*x={M_expected} from preload, but mapping has M={M}. "
    "This FFT approach assumes a full rectangular unmasked grid."
)

def curvature_matrix_from_preload_jax(
    w_preload_np: np.ndarray,
    rows_np: np.ndarray,
    cols_np: np.ndarray,
    vals_np: np.ndarray,
    y_shape: int,
    x_shape: int,
    S: int,
    batch_size: int = 32,
    eps_dtype=np.float64,
):
    import jax
    import jax.numpy as jnp
    from jax.ops import segment_sum

    # -------------------------
    # Move constants to device
    # -------------------------
    w = jnp.array(w_preload_np, dtype=eps_dtype)      # (2y,2x)
    rows = jnp.array(rows_np, dtype=jnp.int32)        # (nnz,)
    cols = jnp.array(cols_np, dtype=jnp.int32)        # (nnz,)
    vals = jnp.array(vals_np, dtype=eps_dtype)        # (nnz,)

    M = y_shape * x_shape
    nnz = rows.shape[0]

    # -------------------------
    # FFT of W-tilde kernel
    # -------------------------
    Khat = jnp.fft.fft2(w)                            # (2y,2x) complex

    def apply_operator_fft_batch(Fbatch_flat):
        """
        Fbatch_flat: (M, B)
        returns:     (M, B)
        """
        B = Fbatch_flat.shape[1]

        F_img = Fbatch_flat.T.reshape((B, y_shape, x_shape))
        F_pad = jnp.pad(F_img, ((0, 0), (0, y_shape), (0, x_shape)))

        Fhat = jnp.fft.fft2(F_pad)
        Ghat = Fhat * Khat[None, :, :]
        G_pad = jnp.fft.ifft2(Ghat)

        G = jnp.real(G_pad[:, :y_shape, :x_shape])
        return G.reshape((B, M)).T                     # (M,B)

    # -------------------------
    # One fixed-size batch kernel
    # -------------------------
    @jax.jit
    def compute_batch(start_col):
        """
        Always returns shape (S, batch_size)
        """
        in_batch = (cols >= start_col) & (cols < start_col + batch_size)

        bc = jnp.where(in_batch, cols - start_col, 0).astype(jnp.int32)
        v = jnp.where(in_batch, vals, 0.0)

        Fbatch = jnp.zeros((M, batch_size), dtype=eps_dtype)
        Fbatch = Fbatch.at[rows, bc].add(v)

        Gbatch = apply_operator_fft_batch(Fbatch)

        G_at_rows = Gbatch[rows, :]
        contrib = vals[:, None] * G_at_rows

        Cbatch = segment_sum(contrib, cols, num_segments=S)
        return Cbatch

    # -------------------------
    # Assemble full C
    # -------------------------
    C = jnp.zeros((S, S), dtype=eps_dtype)

    for start in range(0, S, batch_size):
        Cbatch = compute_batch(start)

        # Handle tail safely outside JIT
        Btail = min(batch_size, S - start)
        C = C.at[:, start:start + Btail].set(Cbatch[:, :Btail])

    # Enforce symmetry
    C = 0.5 * (C + C.T)
    return C



# -------------------------
# Run + compare
# -------------------------
import jax
import jax.numpy as jnp

w_preload_jax = jnp.asarray(w_tilde_preload)
rows_jax = jnp.asarray(rows_np)
cols_jax = jnp.asarray(cols_np)
vals_jax = jnp.asarray(vals_np)


curvature_matrix_from_preload_jax_jit = jax.jit(
    curvature_matrix_from_preload_jax,
    static_argnames=("y_shape", "x_shape", "S", "batch_size"),
)

batch_size = 128

C_jax = curvature_matrix_from_preload_jax_jit(
    w_preload_np=w_preload_jax,
    rows_np=rows_jax,
    cols_np=cols_jax,
    vals_np=vals_jax,
    y_shape=y_shape,
    x_shape=x_shape,
    S=total_mapper_pixels,
    batch_size=batch_size,
)

import time

start = time.time()


C_jax = curvature_matrix_from_preload_jax_jit(
    w_preload_np=w_preload_jax,
    rows_np=rows_jax,
    cols_np=cols_jax,
    vals_np=vals_jax,
    y_shape=y_shape,
    x_shape=x_shape,
    S=total_mapper_pixels,
    batch_size=batch_size,
)

print(C_jax[:5, :5])
print("JAX W TILDE", time.time() - start)

lowered = curvature_matrix_from_preload_jax_jit.lower(
    w_preload_np=w_preload_jax,
    rows_np=rows_jax,
    cols_np=cols_jax,
    vals_np=vals_jax,
    y_shape=y_shape,
    x_shape=x_shape,
    S=total_mapper_pixels,
    batch_size=batch_size,
)
compiled = lowered.compile()
memory_analysis = compiled.memory_analysis()

vram_bytes = (
        memory_analysis.output_size_in_bytes
        + memory_analysis.temp_size_in_bytes
)

print(
        f"VRAM USE = {vram_bytes / 1024 ** 3:.3f} GB"
    )

# C_jax = curvature_matrix_from_preload_jax(
#     w_preload_np=w_tilde_preload,
#     rows_np=rows_np,
#     cols_np=cols_np,
#     vals_np=vals_np,
#     y_shape=y_shape,
#     x_shape=x_shape,
#     S=total_mapper_pixels,
#     batch_size=32,
# )

# Bring back to NumPy for the test
import jax.numpy as jnp
C_jax_np = np.array(C_jax)
#
# print(curvature_matrix_true)
# print(C_jax_np)
# print(C_jax_np - curvature_matrix_true)

# Don't use "==" for floats
assert np.allclose(C_jax_np, curvature_matrix_true, rtol=1e-6, atol=1e-6), (
    f"Max abs diff: {np.max(np.abs(C_jax_np - curvature_matrix_true))}"
)

print("OK: curvature_matrix matches curvature_matrix_true (within tolerance).")



def extract_curvature_for_mask(
    C_rect,
    rect_index_for_mask_index,
):
    """
    Extract curvature matrix for an arbitrary mask from a rectangular curvature matrix.

    Parameters
    ----------
    C_rect : array, shape (S_rect, S_rect)
        Curvature matrix computed on the rectangular grid.
    rect_index_for_mask_index : array, shape (S_mask,)
        For each masked pixel index, gives its index in the rectangular grid.

    Returns
    -------
    C_mask : array, shape (S_mask, S_mask)
        Curvature matrix for the arbitrary mask.
    """
    xp = type(C_rect)  # works for np and jnp via duck typing

    idx = rect_index_for_mask_index
    return C_rect[idx[:, None], idx[None, :]]
