import numpy as np

import autoarray as aa

# mask_size = 11
# border_size = 2
# kernel_size = 3
#
# mask = np.full(shape=(mask_size, mask_size), fill_value=True)
#
# mask[border_size:mask_size-border_size, border_size:mask_size-border_size] = False
#
# mask = aa.Mask2D(mask=mask, pixel_scales=1.0)
#
# noise_map_native = np.ones(shape=(mask_size,mask_size))
#
# noise_map_native = aa.Array2D(
#     values=noise_map_native, mask=mask
# ).native
#
# kernel_native = aa.Kernel2D.ones(
#     shape_native=(kernel_size,kernel_size), pixel_scales=1.0
# ).native
#
# native_index_for_slim_index = np.array(noise_map_native.mask.derive_indexes.native_for_slim)

from pathlib import Path

instrument = "hst"

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


@aa.numba_util.jit()
def w_tilde_curvature_imaging_from(
    noise_map_native: np.ndarray, kernel_native: np.ndarray, native_index_for_slim_index
) -> np.ndarray:
    """
    The matrix `w_tilde_curvature` is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF
    convolution of every pair of image pixels given the noise map. This can be used to efficiently compute the
    curvature matrix via the mappings between image and source pixels, in a way that omits having to perform the
    PSF convolution on every individual source pixel. This provides a significant speed up for inversions of imaging
    datasets.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_curvature_preload_imaging_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Parameters
    ----------
    noise_map_native
        The two dimensional masked noise-map of values which w_tilde is computed from.
    kernel_native
        The two dimensional PSF kernel that w_tilde encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the noise map that enables efficient calculation of
        the curvature matrix.
    """
    image_pixels = len(native_index_for_slim_index)

    w_tilde_curvature = np.zeros((image_pixels, image_pixels))

    for ip0 in range(w_tilde_curvature.shape[0]):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        for ip1 in range(ip0, w_tilde_curvature.shape[1]):
            ip1_y, ip1_x = native_index_for_slim_index[ip1]

            w_tilde_curvature[
                ip0, ip1
            ] += aa.util.inversion_imaging_numba.w_tilde_curvature_value_from(
                value_native=noise_map_native,
                kernel_native=kernel_native,
                ip0_y=ip0_y,
                ip0_x=ip0_x,
                ip1_y=ip1_y,
                ip1_x=ip1_x,
            )

    for ip0 in range(w_tilde_curvature.shape[0]):
        for ip1 in range(ip0, w_tilde_curvature.shape[1]):
            w_tilde_curvature[ip1, ip0] = w_tilde_curvature[ip0, ip1]

    return w_tilde_curvature


@aa.numba_util.jit()
def w_tilde_curvature_preload_imaging_from(
    noise_map_native: np.ndarray, kernel_native: np.ndarray, native_index_for_slim_index
):
    """
    The matrix `w_tilde_curvature` is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF
    convolution of every pair of image pixels on the noise map. This can be used to efficiently compute the
    curvature matrix via the mappings between image and source pixels, in a way that omits having to repeat the PSF
    convolution on every individual source pixel. This provides a significant speed up for inversions of imaging
    datasets.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations slow. This methods creates
    a sparse matrix that can compute the matrix `w_tilde_curvature` efficiently, albeit the linear algebra calculations
    in PyAutoArray bypass this matrix entirely to go straight to the curvature matrix.

    for dataset data, w_tilde is a sparse matrix, whereby non-zero entries are only contained for pairs of image pixels
    where the two pixels overlap due to the kernel size. For example, if the kernel size is (11, 11) and two image
    pixels are separated by more than 20 pixels, the kernel will never convolve flux between the two pixels. Two image
    pixels will only share a convolution if they are within `kernel_overlap_size = 2 * kernel_shape - 1` pixels within
    one another.

    Thus, a `w_tilde_curvature_preload` matrix of dimensions [image_pixels, kernel_overlap_size ** 2] can be computed
    which significantly reduces the memory consumption by removing the sparsity. Because the dimensions of the second
    axes is no longer `image_pixels`, a second matrix `w_tilde_indexes` must also be computed containing the slim image
    pixel indexes of every entry of `w_tilde_preload`.

    In order for the preload to store half the number of values, owing to the symmetry of the `w_tilde_curvature`
    matrix, the image pixel pairs corresponding to the same image pixel are divided by two. This ensures that when the
    curvature matrix is computed these pixels are not double-counted.

    The values stored in `w_tilde_curvature_preload` represent the convolution of overlapping noise-maps given the
    PSF kernel. It is common for many values to be neglibly small. Removing these values can speed up the inversion
    and reduce memory at the expense of a numerically irrelevent change of solution.

    This matrix can then be used to compute the `curvature_matrix` in a memory efficient way that exploits the sparsity
    of the linear algebra.

    Parameters
    ----------
    noise_map_native
        The two dimensional masked noise-map of values which `w_tilde_curvature` is computed from.
    signal_to_noise_map_native
        The two dimensional masked signal-to-noise-map from which the threshold discarding low S/N image pixel
        pairs is used.
    kernel_native
        The two dimensional PSF kernel that `w_tilde_curvature` encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the noise map that enables efficient calculation of
        the curvature matrix, where the dimensions are reduced to save memory.
    """

    image_pixels = len(native_index_for_slim_index)

    kernel_overlap_size = (2 * kernel_native.shape[0] - 1) * (
        2 * kernel_native.shape[1] - 1
    )

    curvature_preload_tmp = np.zeros((image_pixels, kernel_overlap_size))
    curvature_indexes_tmp = np.zeros((image_pixels, kernel_overlap_size))
    curvature_lengths = np.zeros(image_pixels)

    for ip0 in range(image_pixels):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        kernel_index = 0

        for ip1 in range(ip0, curvature_preload_tmp.shape[0]):
            ip1_y, ip1_x = native_index_for_slim_index[ip1]

            noise_value = aa.util.inversion_imaging_numba.w_tilde_curvature_value_from(
                value_native=noise_map_native,
                kernel_native=kernel_native,
                ip0_y=ip0_y,
                ip0_x=ip0_x,
                ip1_y=ip1_y,
                ip1_x=ip1_x,
            )

            if ip0 == ip1:
                noise_value /= 2.0

            if noise_value > 0.0:
                curvature_preload_tmp[ip0, kernel_index] = noise_value
                curvature_indexes_tmp[ip0, kernel_index] = ip1
                kernel_index += 1

        curvature_lengths[ip0] = kernel_index

    curvature_total_pairs = int(np.sum(curvature_lengths))

    curvature_preload = np.zeros((curvature_total_pairs))
    curvature_indexes = np.zeros((curvature_total_pairs))

    index = 0

    for i in range(image_pixels):
        for data_index in range(int(curvature_lengths[i])):
            curvature_preload[index] = curvature_preload_tmp[i, data_index]
            curvature_indexes[index] = curvature_indexes_tmp[i, data_index]

            index += 1

    return (curvature_preload, curvature_indexes, curvature_lengths)


w_tilde_curvature = w_tilde_curvature_imaging_from(
    noise_map_native=dataset.noise_map.native.array,
    kernel_native=dataset.psf.native.array,
    native_index_for_slim_index=np.array(mask.derive_indexes.native_for_slim).astype(
        "int"
    ),
)

curvature_preload, curvature_indexes, curvature_lengths = (
    w_tilde_curvature_preload_imaging_from(
        noise_map_native=dataset.noise_map.native.array,
        kernel_native=dataset.psf.native.array,
        native_index_for_slim_index=np.array(
            mask.derive_indexes.native_for_slim
        ).astype("int"),
    )
)


from numba import njit, prange
import numba
import numpy as np

print("Numba is using", numba.get_num_threads(), "threads")


@njit(parallel=True, fastmath=True)
def curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
    curvature_preload,
    curvature_indexes,
    curvature_lengths,
    data_to_pix_unique,
    data_weights,
    pix_lengths,
    pix_pixels,
):
    data_pixels = curvature_lengths.shape[0]
    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    curvature_index = 0

    for data_0 in prange(data_pixels):  # <--- parallelize outer loop
        ci = curvature_index
        for data_1_index in range(curvature_lengths[data_0]):
            data_1 = curvature_indexes[ci]
            w_tilde_value = curvature_preload[ci]

            for pix_0_index in range(pix_lengths[data_0]):
                data_0_weight = data_weights[data_0, pix_0_index]
                pix_0 = data_to_pix_unique[data_0, pix_0_index]

                for pix_1_index in range(pix_lengths[data_1]):
                    data_1_weight = data_weights[data_1, pix_1_index]
                    pix_1 = data_to_pix_unique[data_1, pix_1_index]

                    curvature_matrix[pix_0, pix_1] += (
                        data_0_weight * data_1_weight * w_tilde_value
                    )
            ci += 1

    # symmetrize
    for i in prange(pix_pixels):
        for j in range(i, pix_pixels):
            curvature_matrix[i, j] += curvature_matrix[j, i]
            curvature_matrix[j, i] = curvature_matrix[i, j]

    return curvature_matrix


curvature_matrix = curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
    curvature_preload=curvature_preload,
    curvature_indexes=curvature_indexes.astype("int"),
    curvature_lengths=curvature_lengths.astype("int"),
    data_to_pix_unique=np.array(data_to_pix_unique),
    data_weights=np.array(data_weights),
    pix_lengths=np.array(pix_lengths).astype("int"),
    pix_pixels=mapping_matrix.shape[1],
)

import time

start = time.time()

curvature_matrix = curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
    curvature_preload=curvature_preload,
    curvature_indexes=curvature_indexes.astype("int"),
    curvature_lengths=curvature_lengths.astype("int"),
    data_to_pix_unique=np.array(data_to_pix_unique),
    data_weights=np.array(data_weights),
    pix_lengths=np.array(pix_lengths).astype("int"),
    pix_pixels=mapping_matrix.shape[1],
)

print(f"Time numba CPU curvature_matrix calculation via w_tilde: {time.time() - start}")
