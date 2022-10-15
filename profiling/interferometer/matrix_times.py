import autolens as al
import time
import numpy as np
from numba import jit

arr_1d = np.arange(13000)


@jit(nopython=True, cache=True, parallel=False)
def simple_jit(arr_1d,):
    """
    Use the preloaded w_tilde matrix (see `w_tilde_preload_from_jit`) to compute w_tilde efficiently.

    Parameters
    ----------
    w_tilde_preload
        The preloaded values of the NUFFT that enable efficient computation of w_tilde.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.

    Returns
    -------
    w_tilde
        Matrix that makes linear algebra fast.
    """

    arr_2d = np.zeros((arr_1d.shape[0], arr_1d.shape[0]))

    for i in range(arr_1d.shape[0]):
        for j in range(i, arr_1d.shape[0]):

            arr_2d[i, j] = arr_1d[i] + arr_2d[i, j]

    return arr_2d


start = time.time()

arr_2d = simple_jit(arr_1d=arr_1d)

print(f"TIME VIA JIT = {time.time() - start}")

print(arr_2d.shape)


arr_2d = np.dot(arr_1d, arr_1d)
