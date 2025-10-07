import numpy as np
import scipy.sparse as sp
import time

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse


def make_sparse_mapping_coo(
    image_pixels=10000, source_pixels=1000, nnz_per_col=66, seed=0
):
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for col in range(source_pixels):
        r = rng.choice(image_pixels, size=nnz_per_col, replace=False)
        v = rng.random(nnz_per_col)
        rows.append(r)
        cols.append(np.full(nnz_per_col, col))
        vals.append(v)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def make_sparse_wtilde_coo(image_pixels=10000, nnz_per_row=2000, seed=1):
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for i in range(image_pixels):
        c = rng.choice(image_pixels, size=nnz_per_row, replace=False)
        v = rng.random(nnz_per_row)
        rows.append(np.full(nnz_per_row, i))
        cols.append(c)
        vals.append(v)
        # enforce symmetry
        rows.append(c)
        cols.append(np.full(nnz_per_row, i))
        vals.append(v)
    diag = np.arange(image_pixels)
    diag_vals = rng.random(image_pixels) + nnz_per_row
    rows.append(diag)
    cols.append(diag)
    vals.append(diag_vals)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def benchmark(image_pixels=2000, source_pixels=500, nnz_per_col=10, nnz_per_row=200):
    print(f"\nBenchmarking with image={image_pixels}, source={source_pixels}")

    # Build sparse COO matrices
    M_r, M_c, M_v = make_sparse_mapping_coo(image_pixels, source_pixels, nnz_per_col)
    W_r, W_c, W_v = make_sparse_wtilde_coo(image_pixels, nnz_per_row)

    # Sparse matrices (SciPy CPU)
    M_sparse = sp.coo_matrix((M_v, (M_r, M_c)), shape=(image_pixels, source_pixels))
    W_sparse = sp.coo_matrix(
        (W_v, (W_r, W_c)), shape=(image_pixels, image_pixels)
    ).tocsr()

    # Dense matrices (for reference — very costly for large image_pixels!)
    M_dense = M_sparse.toarray()
    W_dense = W_sparse.toarray()
    print(
        f"Mapping Matrix non zero per column {np.count_nonzero(W_dense, axis=0).mean()}"
    )
    print(f"Mapping Matrix non zero per row {np.count_nonzero(W_dense, axis=1).mean()}")

    # --- Dense NumPy ---
    t0 = time.time()
    F_dense = M_dense.T @ (W_dense @ M_dense)
    t1 = time.time()
    print(f"Dense time: {t1 - t0:.3f} s | F_dense shape {F_dense.shape}")

    # --- Sparse SciPy ---
    t0 = time.time()
    M_sparse = M_sparse.tocsr()
    F_sparse = M_sparse.T @ (W_sparse @ M_sparse)  # result is sparse
    t1 = time.time()
    print(
        f"Sparse SciPy time: {t1 - t0:.3f} s | F_sparse shape {F_sparse.shape} | nnz={F_sparse.nnz}"
    )

    # --- Sparse CuPy (GPU) ---
    M_cu = cpx_sparse.coo_matrix(
        (cp.asarray(M_v), (cp.asarray(M_r), cp.asarray(M_c))),
        shape=(image_pixels, source_pixels),
    ).tocsr()
    W_cu = cpx_sparse.coo_matrix(
        (cp.asarray(W_v), (cp.asarray(W_r), cp.asarray(W_c))),
        shape=(image_pixels, image_pixels),
    ).tocsr()

    # Warm-up (to avoid counting kernel compilation)
    _ = M_cu.T @ (W_cu @ M_cu)
    cp.cuda.Stream.null.synchronize()

    t0 = time.time()
    F_cu = M_cu.T @ (W_cu @ M_cu)
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()
    print(
        f"Sparse CuPy (GPU) time: {t1 - t0:.3f} s | F_cu shape {F_cu.shape} | nnz={F_cu.nnz}"
    )

    # Warm-up (to avoid counting kernel compilation)
    _ = M_cu.T @ (W_cu @ M_cu)
    cp.cuda.Stream.null.synchronize()

    t0 = time.time()
    F_cu = M_cu.T @ (W_cu @ M_cu)
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()
    print(
        f"Sparse CuPy (GPU) time: {t1 - t0:.3f} s | F_cu shape {F_cu.shape} | nnz={F_cu.nnz}"
    )

    # --- Validate results ---
    t0 = time.time()
    F_sparse_arr = F_sparse.toarray()
    F_cu_arr = F_cu.toarray().get()
    t1 = time.time()
    print(f"Transfer to dense and CPU time: {t1 - t0:.3f} s")

    assert np.allclose(
        F_sparse_arr, F_cu_arr, rtol=1e-5, atol=1e-8
    ), "Mismatch between SciPy and CuPy results!"
    print("✅ CuPy and SciPy results match within tolerance.")


if __name__ == "__main__":

    # Larger case (careful with dense memory!)
    benchmark(image_pixels=11304, source_pixels=1024, nnz_per_col=66, nnz_per_row=730)
