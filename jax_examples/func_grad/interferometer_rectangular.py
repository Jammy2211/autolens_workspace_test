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
import jax
from jax import grad
from os import path

import autofit as af
import autolens as al
from autoconf import conf

"""
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=4.0
)

"""
__Dataset__

Load and plot the galaxy dataset `operated` via .fits files, which we will fit with 
the model.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)
#
# dataset = al.Interferometer.from_fits(
#     data_path=path.join(dataset_path, "data.fits"),
#     noise_map_path=path.join(dataset_path, "noise_map.fits"),
#     uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
#     real_space_mask=real_space_mask,
#     transformer_class=al.TransformerDFT,
# )

total_visibilities = 50000

data = al.Visibilities(np.random.normal(loc=0.0, scale=1.0, size=total_visibilities) + 1j * np.random.normal(
    loc=0.0, scale=1.0, size=total_visibilities
))

noise_map = al.VisibilitiesNoiseMap(np.ones(total_visibilities) + 1j * np.ones(total_visibilities))

uv_wavelengths = np.random.uniform(
    low=-300.0, high=300.0, size=(total_visibilities, 2)
)

dataset = al.Interferometer(
    data=data,
    noise_map=noise_map,
    uv_wavelengths=uv_wavelengths,
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

print(f"Total Visiblities: {dataset.uv_wavelengths.shape[0]}")

"""
__Mask__

The model-fit requires a 2D mask defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)

# over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
#     grid=dataset.grid,
#     sub_size_list=[4, 2, 1],
#     radial_list=[0.3, 0.6],
#     centre_list=[(0.0, 0.0)],
# )
#
# dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__JAX & Preloads__

In JAX, calculations must use static shaped arrays with known and fixed indexes. For certain calculations in the
pixelization, this information has to be passed in before the pixelization is performed. Below, we do this for 3
inputs:

- `total_linear_light_profiles`: The number of linear light profiles in the model. This is 0 because we are not
  fitting any linear light profiles to the data, primarily because the lens light is omitted.

- `total_mapper_pixels`: The number of source pixels in the rectangular pixelization mesh. This is required to set up 
  the arrays that perform the linear algebra of the pixelization.

- `source_pixel_zeroed_indices`: The indices of source pixels on its edge, which when the source is reconstructed 
  are forced to values of zero, a technique tests have shown are required to give accruate lens models.

The `image_mesh` can be ignored, it is legacy API from previous versions which may or may not be reintegrated in future
versions.
"""
image_mesh = None
mesh_shape = (30, 30)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

preloads = al.Preloads(
    mapper_indices=al.mapper_indices_from(
        total_linear_light_profiles=0,
        total_mapper_pixels=total_mapper_pixels
    ),
    source_pixel_zeroed_indices=al.util.mesh.rectangular_edge_pixel_list_from(
        mesh_shape
    ),
)

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a model where:

 - The galaxy's bulge is a parametric `Sersic` bulge [7 parameters]. 
 - The galaxy's point source emission is a parametric operated `Gaussian` centred on the bulge [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
# # Lens:

mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)
mass.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=0.11111111111111108, upper_limit=0.1111111111111111
)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)
shear.gamma_2 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=mass,
    shear=shear,
)

# Source:

mesh = al.mesh.RectangularMagnification(shape=mesh_shape)
# regularization = al.reg.Constant(coefficient=1.0)

# regularization = al.reg.GaussianKernel(coefficient=1.0, scale=1.0)

regularization = al.reg.AdaptiveBrightness()

pixelization = al.Pixelization(
    mesh=mesh, regularization=regularization
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))



bulge = al.lp.Sersic()

image = bulge.image_2d_from(grid=dataset.grid)

galaxy_name_image_dict = {
    "('galaxies', 'lens')": image,
    "('galaxies', 'source')": image
}

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
can compute its gradient.
"""
import jax.numpy as jnp

analysis = al.AnalysisInterferometer(
    dataset=dataset,
    #    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    preloads=preloads,
    raise_inversion_positions_likelihood_exception=False,
)

analysis._adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_name_image_dict
)

"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**, which pairs
the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
from autofit.non_linear.fitness import Fitness
import time

use_vmap = False
batch_size = 5

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


"""
Output an image of the fit, so that we can inspect that it fits the data as expected.
"""
import autolens.plot as aplt
import os

file_path = os.path.join(al.__version__)

instance = model.instance_from_prior_medians()

fit = analysis.fit_from(instance)

print(f"Figure of Merit = {fit.figure_of_merit}")

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"interferometer_subplot_fit", format="png"
    )
)
fit_plotter = aplt.FitInterferometerPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_fit()


mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"interferometer_subplot_inversion_0", format="png"
    )
)
fit_plotter = aplt.InversionPlotter(inversion=fit.inversion, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_of_mapper(mapper_index=0)
