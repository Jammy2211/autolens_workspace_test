import jax.numpy as jnp
from pathlib import Path
import numpy as np
import time

import autoarray as aa

# instrument = "vro"
# instrument = "euclid"
instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

folder = Path("linear_alg") / "arrs" / instrument

data_to_pix_unique = np.load(f"{folder}/data_to_pix_unique.npy")
data_weights = np.load(f"{folder}/data_weights.npy")
pix_lengths = np.load(f"{folder}/pix_lengths.npy")
mapping_matrix = np.load(f"{folder}/mapping_matrix.npy")
curvature_matrix = np.load(f"{folder}/curvature_matrix.npy")

pixel_scales_dict = {"vro": 0.2, "euclid": 0.1, "hst": 0.05, "hst_up": 0.03, "ao": 0.01}
pixel_scale = pixel_scales_dict[instrument]

dataset_path = Path("dataset") / "imaging" / "instruments" / instrument

dataset = aa.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
    over_sample_size_pixelization=4,
)

mask = aa.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

"""
__Mask__
"""
dataset = dataset.apply_mask(mask=mask)

"""
Precompute functions so compute tile not printed
"""
from autoarray.inversion.inversion import inversion_util
from autoarray.inversion.inversion.imaging import inversion_imaging_numba_util

# blurred_mapping_matrix_calc = dataset.convolver.convolve_mapping_matrix(
#     mapping_matrix=mapping_matrix
# )
#
# curvature_matrix_calc = inversion_util.curvature_matrix_via_mapping_matrix_from(
#     mapping_matrix=blurred_mapping_matrix_calc, noise_map=dataset.noise_map
# )

curvature_matrix_w_tilde = inversion_imaging_numba_util.curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
    curvature_preload=dataset.w_tilde.curvature_preload,
    curvature_indexes=dataset.w_tilde.indexes,
    curvature_lengths=dataset.w_tilde.lengths,
    data_to_pix_unique=np.array(data_to_pix_unique),
    data_weights=np.array(data_weights),
    pix_lengths=np.array(pix_lengths),
    pix_pixels=mapping_matrix.shape[1],
)

"""
__Shapes__
"""
print(f"Data Shape: {dataset.data.shape_native}")
print(f"PSF Shape: {dataset.psf.shape_native}")
print(f"Mapping Matrix Shape: {mapping_matrix.shape}")
print("Curvature Matrix Shape: ", curvature_matrix.shape)

# print(f"Mapping Matrix non zero per column {np.count_nonzero(mapping_matrix, axis=0).mean()}")
# print(f"Mapping Matrix non zero per row {np.count_nonzero(mapping_matrix, axis=1).mean()}")
# print(f"W Matrix non zero per column {np.count_nonzero(dataset.w_tilde.w_matrix, axis=0).mean()}")
# print(f"W Matrix non zero per row {np.count_nonzero(dataset.w_tilde.w_matrix, axis=1).mean()}")
# ffff

"""
__Time__
"""
# start = time.time()
#
# blurred_mapping_matrix_calc = dataset.convolver.convolve_mapping_matrix(
#     mapping_matrix=mapping_matrix
# )
#
# print(f"Time numba CPU blurred mapping matrix calculation: {time.time() - start}")
#
# start = time.time()
#
# curvature_matrix_calc = inversion_util.curvature_matrix_via_mapping_matrix_from(
#     mapping_matrix=blurred_mapping_matrix_calc, noise_map=dataset.noise_map
# )
#
# print(
#     f"Time numba CPU curvature_matrix calculation via mapping matrix: {time.time() - start}"
# )

start = time.time()

curvature_matrix_w_tilde = inversion_imaging_numba_util.curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
    curvature_preload=dataset.w_tilde.curvature_preload,
    curvature_indexes=dataset.w_tilde.indexes,
    curvature_lengths=dataset.w_tilde.lengths,
    data_to_pix_unique=np.array(data_to_pix_unique),
    data_weights=np.array(data_weights),
    pix_lengths=np.array(pix_lengths),
    pix_pixels=mapping_matrix.shape[1],
)

print(f"Time numba CPU curvature_matrix calculation via w_tilde: {time.time() - start}")


import time
import numpy as np
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse


def benchmark_with_real_data(mapping_matrix, w_matrix, F_numba):
    """
    Benchmark curvature matrix construction using real saved arrays.
    mapping_matrix : np.ndarray [image_pixels, source_pixels]
    w_matrix       : np.ndarray [image_pixels, image_pixels]
    """
    image_pixels, source_pixels = mapping_matrix.shape
    print(f"\nBenchmarking with image={image_pixels}, source={source_pixels}")

    # --- Convert to SciPy sparse ---
    print("Building sparse matrices...")
    t0 = time.time()
    M_sparse = sp.csr_matrix(mapping_matrix)
    W_sparse = sp.csr_matrix(w_matrix)
    t1 = time.time()
    print(
        f" SciPy sparse build: {t1 - t0:.3f}s | nnz M={M_sparse.nnz}, W={W_sparse.nnz}"
    )

    # --- SciPy Sparse Benchmark ---
    t0 = time.time()
    F_sparse = M_sparse.T @ (W_sparse @ M_sparse)
    t1 = time.time()
    print(
        f" SciPy sparse compute: {t1 - t0:.3f}s | shape={F_sparse.shape} | nnz={F_sparse.nnz}"
    )

    # --- CuPy Sparse Benchmark (GPU) ---
    print("Sending to GPU...")
    t0 = time.time()
    M_cu = cpx_sparse.csr_matrix(cp.asarray(mapping_matrix))
    W_cu = cpx_sparse.csr_matrix(cp.asarray(w_matrix))
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()
    print(f" CuPy sparse build: {t1 - t0:.3f}s | nnz M={M_cu.nnz}, W={W_cu.nnz}")

    F_cu = M_cu.T @ (W_cu @ M_cu)

    t0 = time.time()
    F_cu = M_cu.T @ (W_cu @ M_cu)
    cp.cuda.Stream.null.synchronize()  # ensure timing is accurate
    t1 = time.time()
    print(f" CuPy sparse compute: {t1 - t0:.3f}s | shape={F_cu.shape} | nnz={F_cu.nnz}")

    # --- Norm check (Frobenius norm of data only) ---
    fro_scipy = np.sqrt((F_sparse.data**2).sum())
    fro_cupy = cp.sqrt(cp.sum(F_cu.data**2)).get()
    print(
        f" Norm check: SciPy={fro_scipy:.4e}, CuPy={fro_cupy:.4e}, diff={abs(fro_scipy - fro_cupy):.4e}"
    )

    # Frobenius norm difference
    diff = np.linalg.norm(F_numba - F_cu.toarray().get())
    print(f"F-norm difference: {diff:.3e}")
    assert diff < 1e-4

    # Transfer to NumPy
    t0 = time.time()
    F_numpy = F_cu.toarray().get()  # GPU → CPU
    t1 = time.time()
    print(f" Transfer GPU→CPU: {t1 - t0:.3f} s")

    import jax.dlpack

    # Direct transfer CuPy → JAX on GPU
    t0 = time.time()
    F_jax = jax.dlpack.from_dlpack(F_cu.toDlpack())
    cp.cuda.Stream.null.synchronize()  # ensure kernel + transfer finished
    t1 = time.time()
    print(f" Transfer CuPy→JAX (DLPack, zero-copy): {t1 - t0:.3f} s")

    return F_sparse, F_cu


F_sparse, F_cu = benchmark_with_real_data(
    mapping_matrix, dataset.w_tilde.w_matrix, curvature_matrix_w_tilde
)
