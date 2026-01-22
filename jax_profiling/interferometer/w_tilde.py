import jax
import jax.numpy as jnp
import numpy as np
import time

import autolens as al

mask_radius = 1.0

real_space_mask = al.Mask2D.circular(
    shape_native=(80, 80), pixel_scales=0.05, radius=mask_radius
)

grid = al.Grid2D.from_mask(real_space_mask)


total_visibilities = 10000

data = al.Visibilities(np.random.normal(loc=0.0, scale=1.0, size=total_visibilities) + 1j * np.random.normal(
    loc=0.0, scale=1.0, size=total_visibilities
))

noise_map = np.real(np.ones(total_visibilities) + 1j * np.ones(total_visibilities))

uv_wavelengths = np.random.uniform(
    low=-300.0, high=300.0, size=(total_visibilities, 2)
)

@jax.jit
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    grid_radians_slim: np.ndarray[tuple[int, int], np.float64],
) -> np.ndarray[tuple[int, int], np.float64]:
    r"""
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Note that the current implementation does not take advantage of the fact that w_tilde is symmetric,
    due to the use of vectorized operations.

    .. math::
        \tilde{W}_{ij} = \sum_{k=1}^N \frac{1}{n_k^2} \cos(2\pi[(g_{i1} - g_{j1})u_{k0} + (g_{i0} - g_{j0})u_{k1}])

    The function is written in a way that the memory use does not depend on size of data K.

    Parameters
    ----------
    noise_map_real : ndarray, shape (K,), dtype=float64
        The real noise-map values of the interferometer data.
    uv_wavelengths : ndarray, shape (K, 2), dtype=float64
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim : ndarray, shape (M, 2), dtype=float64
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    curvature_matrix : ndarray, shape (M, M), dtype=float64
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    TWO_PI = 2.0 * jnp.pi

    M = grid_radians_slim.shape[0]
    g_2pi = TWO_PI * grid_radians_slim
    δg_2pi = g_2pi.reshape(M, 1, 2) - g_2pi.reshape(1, M, 2)
    δg_2pi_y = δg_2pi[:, :, 0]
    δg_2pi_x = δg_2pi[:, :, 1]

    def f_k(
        noise_map_real: float,
        uv_wavelengths: np.ndarray[tuple[int], np.float64],
    ) -> np.ndarray[tuple[int, int], np.float64]:
        return jnp.cos(δg_2pi_x * uv_wavelengths[0] + δg_2pi_y * uv_wavelengths[1]) * jnp.reciprocal(
            jnp.square(noise_map_real)
        )

    def f_scan(
        sum_: np.ndarray[tuple[int, int], np.float64],
        args: tuple[float, np.ndarray[tuple[int], np.float64]],
    ) -> tuple[np.ndarray[tuple[int, int], np.float64], None]:
        noise_map_real, uv_wavelengths = args
        return sum_ + f_k(noise_map_real, uv_wavelengths), None

    res, _ = jax.lax.scan(
        f_scan,
        jnp.zeros((M, M)),
        (
            noise_map_real,
            uv_wavelengths,
        ),
    )
    return res

##### JAX ####

w_tilde_jax = w_tilde_curvature_interferometer_from(
    noise_map_real=noise_map,
    uv_wavelengths=uv_wavelengths,
    grid_radians_slim=grid.in_radians.array,
)
print(w_tilde_jax[0,0])

start = time.time()

w_tilde_jax = w_tilde_curvature_interferometer_from(
    noise_map_real=noise_map,
    uv_wavelengths=uv_wavelengths,
    grid_radians_slim=grid.in_radians.array,
)
print(w_tilde_jax[0,0])

print("JAX W TILDE", time.time() - start)



##### NUMBA #####

w_tilde = al.util.inversion_interferometer_numba.w_tilde_curvature_interferometer_from(
    noise_map_real=noise_map,
    uv_wavelengths=uv_wavelengths,
    grid_radians_slim=grid.in_radians.array,
)

start = time.time()

w_tilde = al.util.inversion_interferometer_numba.w_tilde_curvature_interferometer_from(
    noise_map_real=noise_map,
    uv_wavelengths=uv_wavelengths,
    grid_radians_slim=grid.in_radians.array,
)

print("NUMBA W TILDE", time.time() - start)

print(np.max(w_tilde - w_tilde_jax))
