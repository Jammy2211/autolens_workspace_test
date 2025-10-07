import jax.numpy as jnp
import jax
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

w_matrix = np.load(f"{folder}/w_matrix.npy")
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
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
)

"""
Reshape Dataset so that its exactly paired to the extent PSF convolution goes over including the blurring mask edge.

This speeds up JAX calculations as the PSF convolution is done on a smaller array with fewer zero entries.

This will be put in the source code soon during `apply_mask`.
"""


def false_span(mask: np.ndarray):
    """
    Given a boolean mask with False marking valid pixels,
    return the (y_min, y_max), (x_min, x_max) spans of False entries.
    """
    # Find coordinates of False pixels
    ys, xs = np.where(~mask)

    if ys.size == 0 or xs.size == 0:
        raise ValueError("No False entries in mask!")

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    return (y_max - y_min, x_max - x_min)


y_distance, x_distance = false_span(mask=mask.mask)

(pad_y, pad_x) = dataset.psf.shape_native

new_shape = (y_distance + pad_y, x_distance + pad_x)

mask = mask.resized_from(new_shape=new_shape)
data = dataset.data.resized_from(new_shape=new_shape)
noise_map = dataset.noise_map.resized_from(new_shape=new_shape)

dataset = aa.Imaging(
    data=data,
    noise_map=noise_map,
    psf=dataset.psf,
    over_sample_size_pixelization=4,
)

dataset = dataset.apply_mask(mask=mask)


"""
__Shapes__
"""
print(f"Data Shape: {dataset.data.shape_native}")
print(f"PSF Shape: {dataset.psf.shape_native}")
print(f"Mapping Matrix Shape: {mapping_matrix.shape}")
print("Curvature Matrix Shape: ", curvature_matrix.shape)

import via_jax_spike_util as util

curvature_preload_tmp, curvature_indexes, curvature_lengths = (
    util.w_tilde_curvature_preload_imaging_from(
        noise_map_native=dataset.noise_map.native.array,
        kernel_native=dataset.psf.native.array,
        native_index_for_slim_index=np.array(
            mask.derive_indexes.native_for_slim
        ).astype("int"),
    )
)

print(curvature_preload_tmp.shape)
print(curvature_indexes.shape)
print(
    curvature_preload_tmp[0, :]
)  # image pixel zero, all shared kernel blurring entries
print(
    curvature_indexes[0, :]
)  # image pixel zero, all indexes of shared kernel blurring entries


def mapping_matrix_to_graph(mapping_matrix):
    """
    Convert a sparse mapping matrix into a graph-like adjacency list.
    Each row (image pixel) has a list of (source_index, value) pairs for nonzeros.
    """
    rows, cols = np.nonzero(mapping_matrix)
    values = mapping_matrix[rows, cols]

    n_img = mapping_matrix.shape[0]
    adjacency = [[] for _ in range(n_img)]

    for r, c, v in zip(rows, cols, values):
        adjacency[r].append((c, v))

    return adjacency


mapping_matrix_graph = mapping_matrix_to_graph(mapping_matrix)


def curvature_from_graph(curvature_preload_tmp, curvature_indexes, graph, n_src):
    """
    Build curvature matrix using sparse graph form of mapping_matrix.

    Parameters
    ----------
    curvature_preload_tmp : (n_img, n_neighbors)
        PSF weights for each image pixel's neighbors.
    curvature_indexes : (n_img, n_neighbors)
        Neighbor image pixel indices for each image pixel.
    graph : list of lists
        graph[i] = [(src_j, val), ...] nonzero source pixels for image pixel i.
    n_src : int
        Number of source pixels.
    """
    C = np.zeros((n_src, n_src))

    n_img, n_neighbors = curvature_preload_tmp.shape

    for i in range(n_img):
        # Build blurred_row as sparse dict
        blurred_row = {}
        for k in range(n_neighbors):
            w = curvature_preload_tmp[i, k]
            j = curvature_indexes[i, k]
            for src_j, val in graph[j]:  # neighbors' mapping nonzeros
                blurred_row[src_j] = blurred_row.get(src_j, 0.0) + w * val

        # Outer product: only over nonzero entries in M[i] and blurred_row
        for src_j, val_j in graph[i]:  # nonzeros in row i
            for src_k, val_k in blurred_row.items():
                C[src_j, src_k] += val_j * val_k

    return C


"""
Precompute functions so compute tile not printed.
"""
import time

start = time.time()

jtted_func = jax.jit(curvature_from_graph)

print("Precompute time:", time.time() - start)

"""
__Time JIT__
"""
start = time.time()

curvature_matrix_w_tilde = curvature_from_graph(
    curvature_preload_tmp,
    curvature_indexes,
    mapping_matrix_graph,
    mapping_matrix.shape[1],
)

print(f"Time JAX jit curvature_matrix w_tilde: {time.time() - start}")
print(np.max(curvature_matrix_w_tilde))
print(np.min(curvature_matrix_w_tilde))


"""
__Time VMap__
"""
