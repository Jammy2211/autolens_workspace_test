"""
Func Grad: Light Parametric Operated
====================================

This script test if JAX can successfully compute the gradient of the log likelihood of an `Imaging` dataset with a
model which uses operated light profiles.

 __Operated Fitting__

It is common for galaxies to have point-source emission, for example bright emission right at their centre due to
an active galactic nuclei or very compact knot of star formation.

This point-source emission is subject to blurring during data accquisiton due to the telescope optics, and therefore
is not seen as a single pixel of light but spread over multiple pixels as a convolution with the telescope
Point Spread Function (PSF).

It is difficult to model this compact point source emission using a point-source light profile (or an extremely
compact Gaussian / Sersic profile). This is because when the model-image of a compact point source of light is
convolved with the PSF, the solution to this convolution is extremely sensitive to which pixel (and sub-pixel) the
compact model emission lands in.

Operated light profiles offer an alternative approach, whereby the light profile is assumed to have already been
convolved with the PSF. This operated light profile is then fitted directly to the point-source emission, which as
discussed above shows the PSF features.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import jax.numpy as jnp
import jax
from jax import grad
from os import path

import autofit as af
import autolens as al
from autoconf import conf

conf.instance["general"]["model"]["ignore_prior_limits"] = True

"""
__Dataset__

Load and plot the galaxy dataset `operated` via .fits files, which we will fit with 
the model.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)


"""
__Mask__

The model-fit requires a 2D mask defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.
"""
mask_radius = 9.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=mask_radius
)

dataset = dataset.apply_mask(mask=mask)

print(dataset.grids.lp.over_sampled.shape)
print(dataset.grids.pixelization.over_sampled.shape)
print(dataset.grids.blurring.over_sampled.shape)

# dataset = dataset.apply_over_sampling(over_sample_size_lp=1)

# positions = al.Grid2DIrregular(
#     al.from_json(file_path=path.join(dataset_path, "positions.json"))
# )

# over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
#     grid=dataset.grid,
#     sub_size_list=[4, 2, 1],
#     radial_list=[0.3, 0.6],
#     centre_list=[(0.0, 0.0)],
# )
#
# dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)
#

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a model where:

 - The galaxy's bulge is a parametric `Sersic` bulge [7 parameters]. 
 - The galaxy's point source emission is a parametric operated `Gaussian` centred on the bulge [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)

mass = af.Model(al.mp.Isothermal)
mass_0 = af.Model(al.mp.dPIEMass)

mass_0.ell_comps = (0.1, 0.1)
mass_0.b0 = 0.0001
mass_0.rs = 5.0
mass_0.ra = 1.0

shear = af.Model(al.mp.ExternalShear)


lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, mass_0=mass_0, shear=shear)

# Source:

bulge = af.Model(al.lp_linear.Sersic)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
can compute its gradient.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    #   settings_inversion=al.SettingsInversion(use_positive_only_solver=False)
)


"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**, which pairs
the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
from autofit.non_linear.fitness import Fitness
import time

use_vmap = False
batch_size = 50

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

param_vector = jnp.array(model.physical_values_from_prior_medians)

if not use_vmap:

    start = time.time()
    print()
    print(fitness._jit(param_vector))
    print("JAX Time To JIT Function:", time.time() - start)

    start = time.time()
    print()
    print(fitness._jit(param_vector))
    print("JAX Time taken using JIT:", time.time() - start)

else:

    parameters = np.zeros((batch_size, model.total_free_parameters))

    for i in range(batch_size):
        parameters[i, :] = model.physical_values_from_prior_medians

    parameters = jnp.array(parameters)

    start = time.time()
    print()
    print(fitness._vmap(parameters))
    print("JAX Time To VMAP + JIT Function", time.time() - start)

    start = time.time()
    print()
    print(fitness._vmap(parameters))
    print("JAX Time Taken using VMAP:", time.time() - start)
    print("JAX Time Taken per Likelihood:", (time.time() - start) / batch_size)

