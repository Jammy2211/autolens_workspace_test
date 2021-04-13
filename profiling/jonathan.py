import time
import numba
import numpy as np

"""
Depending on if we're using a super computer, we want two different numba decorators:
If on laptop:
@numba.jit(nopython=True, cache=True, parallel=False)
If on super computer:
@numba.jit(nopython=True, cache=False, parallel=True)
"""
# nopython = True
# nopython = True
# cache = False
# parallel = True
# parallel = False
# def jit(nopython=nopython, cache=cache, parallel=parallel):
#    def wrapper(func):
#        return numba.jit(func, nopython=nopython, cache=cache, parallel=parallel)
#
#    return wrapper
def convolve_mapping_matrix(mapping_matrix):
    return convolve_matrix_jit(
        mapping_matrix=mapping_matrix,
        image_frame_1d_indexes=image_frame_1d_indexes,
        image_frame_1d_kernels=image_frame_1d_kernels,
        image_frame_1d_lengths=image_frame_1d_lengths,
    )


from numba import jit


@jit(nopython=True)
def convolve_matrix_jit(
    mapping_matrix,
    image_frame_1d_indexes,
    image_frame_1d_kernels,
    image_frame_1d_lengths,
):
    blurred_mapping_matrix = np.zeros(mapping_matrix.shape)
    for pixel_1d_index in range(mapping_matrix.shape[1]):
        for image_1d_index in range(mapping_matrix.shape[0]):
            value = mapping_matrix[image_1d_index, pixel_1d_index]
            if value > 0:
                frame_1d_indexes = image_frame_1d_indexes[image_1d_index]
                frame_1d_kernel = image_frame_1d_kernels[image_1d_index]
                frame_1d_length = image_frame_1d_lengths[image_1d_index]
                for kernel_1d_index in range(frame_1d_length):
                    vector_index = frame_1d_indexes[kernel_1d_index]
                    kernel_value = frame_1d_kernel[kernel_1d_index]
                    blurred_mapping_matrix[vector_index, pixel_1d_index] += (
                        value * kernel_value
                    )
    return blurred_mapping_matrix


if __name__ == "__main__":
    matrix_shape = (9, 3)
    matrix = np.ones(matrix_shape)
    # iters = 1000000
    iters = 1
    mapping_matrix = np.loadtxt("mapping_matrix.numpy.gz")
    image_frame_1d_indexes = np.loadtxt("image_frame_1d_indexes.numpy.gz", dtype=np.int)
    image_frame_1d_kernels = np.loadtxt("image_frame_1d_kernels.numpy.gz")
    image_frame_1d_lengths = np.loadtxt("image_frame_1d_lengths.numpy.gz")
    print(mapping_matrix.dtype)
    print(image_frame_1d_indexes.dtype)
    print(image_frame_1d_kernels.dtype)
    print(image_frame_1d_lengths.dtype)
    convolve_matrix_jit(
        matrix, image_frame_1d_indexes, image_frame_1d_kernels, image_frame_1d_lengths
    )
    start = time.time()
    for i in range(iters):
        # convolve_mapping_matrix(matrix)
        convolve_matrix_jit(
            matrix,
            image_frame_1d_indexes,
            image_frame_1d_kernels,
            image_frame_1d_lengths,
        )
    end = time.time()
    print(f"time: {end-start}")
