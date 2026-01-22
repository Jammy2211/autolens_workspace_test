import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial

import autolens as al

# Setup test arrays

mask_radius = 0.1

real_space_mask = al.Mask2D.circular(
    shape_native=(80, 80), pixel_scales=0.05, radius=mask_radius
)

grid = al.Grid2D.from_mask(real_space_mask)

total_visibilities = 100

data = al.Visibilities(np.random.normal(loc=0.0, scale=1.0, size=total_visibilities) + 1j * np.random.normal(
    loc=0.0, scale=1.0, size=total_visibilities
))

noise_map = np.real(np.ones(total_visibilities) + 1j * np.ones(total_visibilities))

uv_wavelengths = np.random.uniform(
    low=-300.0, high=300.0, size=(total_visibilities, 2)
)

# Perform JAX calculation

@partial(jax.jit, static_argnums=0)
def w_compact_curvature_interferometer_from(
    grid_size: int,
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    pixel_scale: float,
) -> np.ndarray[tuple[int, int], np.float64]:
    N = grid_size
    OFFSET = N - 1
    # no. of elements after taking the difference of a point in a grid to another
    N_DIFF = 2 * N - 1
    # This converts from arcsec to radian too
    TWOPI_D = (jnp.pi * jnp.pi * pixel_scale) / 324000.0

    δ_mn0 = (TWOPI_D * jnp.arange(grid_size, dtype=jnp.float64)).reshape(-1, 1)
    # shift the centre in the 1-axis
    δ_mn1 = TWOPI_D * (jnp.arange(N_DIFF, dtype=jnp.float64) - OFFSET)

    def f_k(
        noise_map_real: float,
        uv_wavelengths: np.ndarray[tuple[int], np.float64],
    ) -> np.ndarray[tuple[int, int], np.float64]:
        return jnp.cos(δ_mn1 * uv_wavelengths[0] - δ_mn0 * uv_wavelengths[1]) * jnp.square(
            jnp.reciprocal(noise_map_real)
        )

    def f_scan(
        sum_: np.ndarray[tuple[int, int], np.float64],
        args: tuple[float, np.ndarray[tuple[int], np.float64]],
    ) -> tuple[np.ndarray[tuple[int, int], np.float64], None]:
        noise_map_real, uv_wavelengths = args
        return sum_ + f_k(noise_map_real, uv_wavelengths), None

    w_compact, _ = jax.lax.scan(
        f_scan,
        jnp.zeros((N, N_DIFF)),
        (
            noise_map_real,
            uv_wavelengths,
        ),
    )
    return w_compact

grid_size = grid.mask.shape_native_masked_pixels[0] * grid.mask.shape_native_masked_pixels[1]

w_compact = w_compact_curvature_interferometer_from(
    noise_map_real=noise_map,
    uv_wavelengths=uv_wavelengths,
    grid_size=grid_size,
    pixel_scale=grid.pixel_scales[0],
)

@jax.jit
def w_tilde_via_compact_from(
    w_compact: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
) -> np.ndarray[tuple[int, int], np.float64]:
    N = w_compact.shape[0]
    OFFSET = N - 1
    p_ij = native_index_for_slim_index.reshape(-1, 1, 2) - native_index_for_slim_index.reshape(1, -1, 2)
    # flip i, j if first index is negative as cos(-x) = cos(x)
    # this essentially moved the sign of the first index to the second index, and then adds an offset to the second index
    p_ij_1 = jnp.where(jnp.signbit(p_ij[:, :, 0]), -p_ij[:, :, 1], p_ij[:, :, 1]) + OFFSET
    p_ij_0 = jnp.abs(p_ij[:, :, 0])
    return w_compact[p_ij_0, p_ij_1]

w_tilde_jax_compact = w_tilde_via_compact_from(
    w_compact=w_compact,
    native_index_for_slim_index=real_space_mask.derive_indexes.native_for_slim.astype(
        "int"
    ),
)





# compare to numba

w_tilde = al.util.inversion_interferometer.w_tilde_curvature_preload_interferometer_from(
    noise_map_real=noise_map,
    uv_wavelengths=uv_wavelengths,
    shape_masked_pixels_2d=np.array(
        grid.mask.shape_native_masked_pixels
    ),
    grid_radians_2d=np.array(
        grid.mask.derive_grid.all_false.in_radians.native
    ),
)

print(w_tilde.shape)
print(w_tilde_jax_compact.shape)
print(w_tilde - w_tilde_jax_compact)