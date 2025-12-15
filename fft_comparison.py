import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import autolens as al

grid = al.Grid2D.uniform(
    shape_native=(99, 99),
    pixel_scales=0.1,
)

psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=False
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=2.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
)

"""
We now pass these galaxies to a `Tracer`, which performs the ray-tracing calculations they describe and returns
the image of the strong lens system they produce.
"""
tracer = al.Tracer(galaxies=[lens_galaxy])

imaging_jnp = simulator.via_tracer_from(tracer=tracer, grid=grid, xp=jnp)

plt.imshow(imaging_jnp.data.native)
plt.savefig("fft_via_jnp")

imaging_np = simulator.via_tracer_from(tracer=tracer, grid=grid, xp=np)

plt.imshow(imaging_np.data.native)
plt.savefig("fft_via_np")

fft_residual = np.abs(imaging_jnp.data.native - imaging_np.data.native)
plt.imshow(fft_residual)
plt.colorbar()
plt.savefig("fft_residual")