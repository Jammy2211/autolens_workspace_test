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

conf.instance["general"]["model"]["ignore_prior_limits"] = True

sub_size = 4
mask_radius = 3.0
psf_shape_2d = (21, 21)

"""
__Dataset__

Load and plot the galaxy dataset `operated` via .fits files, which we will fit with 
the model.

The simulated data comes at five resolution corresponding to five telescopes:

vro: pixel_scale = 0.2", fastest run times.
euclid: pixel_scale = 0.1", fast run times
hst: pixel_scale = 0.05", normal run times, represents the type of data we do most our fitting on currently.
hst_up: pixel_scale = 0.03", slow run times.
ao: pixel_scale = 0.01", very slow :(
"""
# instrument = "vro"
# instrument = "euclid"
instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

pixel_scales_dict = {"vro": 0.2, "euclid": 0.1, "hst": 0.05, "hst_up": 0.03, "ao": 0.01}
pixel_scale = pixel_scales_dict[instrument]

"""
Load the dataset for this instrument / resolution.
"""
dataset_path = path.join("dataset", "imaging", "instruments", instrument)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scale,
    over_sample_size_lp=sub_size,
    over_sample_size_pixelization=sub_size,
)


"""
__Mask__

The model-fit requires a `Mask2D` defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.
"""
mask_2d = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask_2d)

# dataset = dataset.apply_over_sampling(over_sample_size_lp=1)

# positions = al.Grid2DIrregular(
#     al.from_json(file_path=path.join(dataset_path, "positions.json"))
# )

# over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
#     grid=dataset.grid,
#     sub_size_list=[8, 4, 1],
#     radial_list=[0.3, 0.6],
#     centre_list=[(0.0, 0.0)],
# )
#
# dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)
#
dataset.convolver
dataset.w_tilde.w_matrix
dataset.w_tilde.psf_operator_matrix_dense

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a model where:

 - The galaxy's bulge is a parametric `Sersic` bulge [7 parameters]. 
 - The galaxy's point source emission is a parametric operated `Gaussian` centred on the bulge [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
# # Lens:

# bulge = af.Model(al.lp_linear.Sersic)

bulge = af.Model(al.lp.Sersic)

bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0526316, upper_limit=0.0526318)
bulge.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
bulge.effective_radius = af.UniformPrior(lower_limit=0.5, upper_limit=0.7)
bulge.sersic_index = af.UniformPrior(lower_limit=2.0, upper_limit=4.0)
bulge.intensity = af.UniformPrior(lower_limit=3.0, upper_limit=5.0)

# disk = af.Model(al.lp_linear.Exponential)

disk = af.Model(al.lp.Exponential)

disk.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
disk.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
disk.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.152828012432548, upper_limit=0.1528280124325482)
disk.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.0882352, upper_limit=0.0882353)
disk.effective_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)
disk.intensity = af.UniformPrior(lower_limit=1.0, upper_limit=3.0)

mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.11111111111111108, upper_limit=0.1111111111111111)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(lower_limit=0.000, upper_limit=0.002)
shear.gamma_2 = af.UniformPrior(lower_limit=0.000, upper_limit=0.002)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    disk=disk,
    mass=mass,
    shear=shear,
)

# Source:

mesh_shape = (32, 32)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

mesh = al.mesh.Rectangular(shape=mesh_shape)
# regularization = al.reg.Constant(coefficient=1.0)

regularization = al.reg.GaussianKernel(coefficient=1.0, scale=1.0)

# regularization = al.reg.AdaptiveBrightness()

pixelization = al.Pixelization(
    image_mesh=None, mesh=mesh, regularization=regularization
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

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
import jax.numpy as jnp

analysis = al.AnalysisImaging(
    dataset=dataset,
    #    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    settings_inversion=al.SettingsInversion(
        use_w_tilde=True,
        force_edge_pixels_to_zeros=True,
    ),
    preloads=al.Preloads(
        mapper_indices=al.mapper_indices_from(
            total_linear_light_profiles=0, total_mapper_pixels=total_mapper_pixels
        ),
        source_pixel_zeroed_indices=al.util.mesh.rectangular_edge_pixel_list_from(
            mesh_shape
        ),
    ),
    raise_inversion_positions_likelihood_exception=False,
)

"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**, which pairs
the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
from autofit.non_linear.fitness import Fitness
import time

use_vmap = False
batch_size = 30

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
    print(fitness.call_numpy_wrapper(param_vector))
    print("JAX Time To JIT Function:", time.time() - start)

    start = time.time()
    print()
    print(fitness.call_numpy_wrapper(param_vector))
    print("JAX Time taken using JIT:", time.time() - start)

else:

    parameters = np.zeros((batch_size, model.total_free_parameters))

    for i in range(batch_size):
        parameters[i, :] = model.physical_values_from_prior_medians

    parameters = jnp.array(parameters)

    start = time.time()
    print()
    func = jax.vmap(jax.jit(fitness.call_numpy_wrapper))
    print(func(parameters))
    print("JAX Time To VMAP + JIT Function", time.time() - start)

    start = time.time()
    print()
    print(func(parameters))
    print("JAX Time Taken using VMAP:", time.time() - start)


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
        path=file_path, filename=f"{instrument}_subplot_fit_w_tilde", format="png"
    )
)
fit_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_fit()

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_of_plane_1_w_tilde", format="png"
    )
)
fit_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_of_planes(plane_index=1)

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_inversion_0_w_tilde", format="png"
    )
)
fit_plotter = aplt.InversionPlotter(inversion=fit.inversion, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_of_mapper(mapper_index=0)