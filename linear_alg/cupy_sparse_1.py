import numpy as np
import scipy.sparse as sp
import time

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse


def make_sparse_mapping_coo(
    image_pixels=10000, source_pixels=1000, nnz_per_col=20, seed=0
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


def benchmark(image_pixels=5000, source_pixels=1000, nnz_per_col=20, nnz_per_row=200):
    print(f"\nBenchmarking with image={image_pixels}, source={source_pixels}")

    # Build sparse COO matrices
    M_r, M_c, M_v = make_sparse_mapping_coo(image_pixels, source_pixels, nnz_per_col)
    W_r, W_c, W_v = make_sparse_wtilde_coo(image_pixels, nnz_per_row)

    # --- SciPy CPU ---
    M_sparse = sp.csr_matrix((M_v, (M_r, M_c)), shape=(image_pixels, source_pixels))
    W_sparse = sp.csr_matrix((W_v, (W_r, W_c)), shape=(image_pixels, image_pixels))

    t0 = time.time()
    F_sparse = M_sparse.T @ (W_sparse @ M_sparse)
    t1 = time.time()
    print(
        f"SciPy (CPU) time: {t1 - t0:.3f} s | F shape {F_sparse.shape} | nnz={F_sparse.nnz}"
    )

    # --- CuPy GPU ---
    M_cu = cpx_sparse.csr_matrix(
        (cp.asarray(M_v), (cp.asarray(M_r), cp.asarray(M_c))),
        shape=(image_pixels, source_pixels),
    )
    W_cu = cpx_sparse.csr_matrix(
        (cp.asarray(W_v), (cp.asarray(W_r), cp.asarray(W_c))),
        shape=(image_pixels, image_pixels),
    )

    # warmup (for fair timing, JIT & mempool)
    _ = M_cu.T @ (W_cu @ M_cu)
    cp.cuda.Stream.null.synchronize()

    t0 = time.time()
    F_cu = M_cu.T @ (W_cu @ M_cu)
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()
    print(
        f"CuPy (GPU) time: {t1 - t0:.3f} s | F_cu shape {F_cu.shape} | nnz={F_cu.nnz}"
    )

    # check numerical consistency (small case only)
    if image_pixels < 3000:
        diff = sp.csr_matrix(F_cu.get() - F_sparse)
        print(f"Max abs diff: {np.abs(diff.data).max() if diff.nnz > 0 else 0.0}")


if __name__ == "__main__":
    # Example benchmark (adjust sizes carefully — GPU memory is the limit!)

    benchmark(image_pixels=11304, source_pixels=1024, nnz_per_col=66, nnz_per_row=730)
