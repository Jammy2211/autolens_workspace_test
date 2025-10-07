import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import time


def make_sparse_mapping(image_pixels=2000, source_pixels=500, nnz_per_col=10, seed=0):
    cp.random.seed(seed)
    rows, cols, vals = [], [], []
    for col in range(source_pixels):
        # pick without replacement using permutation
        perm = cp.random.permutation(image_pixels)
        r = perm[:nnz_per_col]
        v = cp.random.random(nnz_per_col)
        rows.append(r)
        cols.append(cp.full(nnz_per_col, col))
        vals.append(v)
    return cp.concatenate(rows), cp.concatenate(cols), cp.concatenate(vals)


def make_sparse_wtilde_upper(image_pixels=2000, nnz_per_row=200, seed=1):
    cp.random.seed(seed)
    rows, cols, vals = [], [], []

    for i in range(image_pixels):
        perm = cp.random.permutation(image_pixels)
        c = perm[:nnz_per_row]
        v = cp.random.random(nnz_per_row)

        mask = c >= i
        if mask.sum() > 0:
            m = int(mask.sum().item())  # convert to Python int
            rows.append(cp.full(m, i))
            cols.append(c[mask])
            vals.append(v[mask])

    diag = cp.arange(image_pixels)
    diag_vals = cp.random.random(image_pixels) + nnz_per_row
    rows.append(diag)
    cols.append(diag)
    vals.append(diag_vals)

    return cp.concatenate(rows), cp.concatenate(cols), cp.concatenate(vals)


def benchmark(image_pixels=2000, source_pixels=500, nnz_per_col=10, nnz_per_row=200):
    print(f"\nCuPy Symmetric Benchmark | image={image_pixels}, source={source_pixels}")

    M_r, M_c, M_v = make_sparse_mapping(image_pixels, source_pixels, nnz_per_col)
    M = cpx_sparse.csr_matrix((M_v, (M_r, M_c)), shape=(image_pixels, source_pixels))

    W_r, W_c, W_v = make_sparse_wtilde_upper(image_pixels, nnz_per_row)
    W_upper = cpx_sparse.csr_matrix(
        (W_v, (W_r, W_c)), shape=(image_pixels, image_pixels)
    )

    # warmup
    F_upper = M.T @ (W_upper @ M)
    F = F_upper + F_upper.T - cpx_sparse.diags(F_upper.diagonal())
    cp.cuda.Stream.null.synchronize()

    # timing
    t0 = time.time()
    F_upper = M.T @ (W_upper @ M)
    F = F_upper + F_upper.T - cpx_sparse.diags(F_upper.diagonal())
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()

    print(
        f"CuPy symmetric F computed in {t1 - t0:.3f} s | shape {F.shape} | nnz={F.nnz}"
    )
    return F


if __name__ == "__main__":
    F = benchmark(
        image_pixels=11304, source_pixels=1024, nnz_per_col=66, nnz_per_row=730
    )
