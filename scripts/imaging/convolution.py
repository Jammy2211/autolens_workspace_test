"""
Modeling: Mass Total + Source Inversion
=======================================

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's surface-brightness is an `Inversion`.

An `Inversion` reconstructs the source's light using a pixel-grid, which is regularized using a prior that forces
this reconstruction to be smooth. This uses `Pixelization`  objects and in this example we will
use their simplest forms, a `RectangularAdaptDensity` `Pixelization` and `Constant` `Regularization`.scheme.

Inversions are covered in detail in chapter 4 of the **HowToLens** lectures.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from os import path
from pathlib import Path

import autolens as al

"""
__Dataset__

Load and plot the strong lens dataset `mass_sie__source_sersic` via .fits files, which we will fit with the lens model.
"""
dataset_label = "build"
dataset_type = "imaging"
dataset_name = "with_lens_light"

dataset_path = path.join("dataset", dataset_label, dataset_type, dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not path.exists(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator/with_lens_light.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)

"""
__Mask__

The model-fit requires a 2D mask defining the regions of the image we fit the lens model to the data, which we define
and use to set up the `Imaging` object that the lens model fits.
"""
mask_radius = 7.2

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
Setup the lens galaxy's mass (SIE+Shear) and source galaxy light (elliptical Sersic) for this simulated lens.

For lens modeling, defining ellipticity in terms of the `ell_comps` improves the model-fitting procedure.

However, for simulating a strong lens you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle, where angle is
in degrees and defined counter clockwise from the positive x-axis.

We can use the `convert` module to determine the elliptical components from the axis-ratio and angle.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.DevVaucouleursSph(
        centre=(0.0, 0.0),
        intensity=0.1,
        effective_radius=0.8,
    ),
    mass=al.mp.IsothermalSph(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.ExponentialSph(
        centre=(0.0, 0.1),
        intensity=0.3,
        effective_radius=0.1,
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

image_2d = tracer.image_2d_from(grid=dataset.grid)

blurring_image_2d = tracer.image_2d_from(
    grid=dataset.grids.blurring,
)


via_fft = dataset.psf.convolved_image_from(
    image=image_2d, blurring_image=blurring_image_2d, xp=jnp
)


via_real_space = dataset.psf.convolved_image_via_real_space_np_from(
    image=image_2d, blurring_image=blurring_image_2d, xp=np
)

residuals = via_fft.native - via_real_space.native

script_path = Path("scripts") / "imaging" / "images"

print(f"Max residual = {residuals.max()}")
print(
    f"Max residual located at {jnp.unravel_index(jnp.argmax(residuals.array), residuals.array.shape)}"
)

plt.imshow(residuals.array, cmap="viridis")
plt.colorbar()
plt.title("Residuals between FFT and Real Space Convolution")
plt.xlabel("X Pixel")
plt.ylabel("Y Pixel")
plt.savefig(script_path / "residuals.png", dpi=300)

mapping_matrix = np.zeros((image_2d.shape[0], 2))

mapping_matrix[:, 0] = image_2d
mapping_matrix[:, 1] = image_2d + 1


via_fft = dataset.psf.convolved_mapping_matrix_from(
    mapping_matrix=mapping_matrix, mask=image_2d.mask, xp=jnp
)


via_real_space = dataset.psf.convolved_mapping_matrix_via_real_space_np_from(
    mapping_matrix=mapping_matrix, mask=image_2d.mask, xp=np
)

residuals = via_fft - via_real_space

print(f"Mapping Matrix Max residual = {residuals.max()}")
