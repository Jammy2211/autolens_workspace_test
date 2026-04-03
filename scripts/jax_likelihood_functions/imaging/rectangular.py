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
from os import path

import autofit as af
import autolens as al

sub_size = 4
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
# instrument = "jwst"
# instrument = "ao"

pixel_scales_dict = {
    "vro": 0.2,
    "euclid": 0.1,
    "hst": 0.05,
    "hst_offset_centre": 0.05,
    "hst_offset_centre_and_mass": 0.05,
    "jwst": 0.03,
    "ao": 0.01,
}
pixel_scale = pixel_scales_dict[instrument]

"""
Load the dataset for this instrument / resolution.
"""
dataset_path = path.join("dataset", "imaging", "instruments", instrument)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not path.exists(dataset_path):
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/imaging/simulator.py"],
        check=True,
    )

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

The model-fit requires a 2D mask defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.
"""
mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

snr_no_lens = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "snr_no_lens.fits"), pixel_scales=pixel_scale
)

signal_to_noise_threshold = 3.0
over_sample_size_pixelization = np.where(
    snr_no_lens.native > signal_to_noise_threshold,
    4,
    2,
)
over_sample_size_pixelization = al.Array2D(
    values=over_sample_size_pixelization, mask=mask
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=over_sample_size_pixelization,
)

# dataset = dataset.apply_sparse_operator(batch_size=64)


"""
__Mesh Shape__

The `mesh_shape` parameter defines number of pixels used by the rectangular mesh to reconstruct the source,
set below to 28 x 28. 

The `mesh_shape` must be fixed before modeling and cannot be a free parameter of the model, because JAX uses the
mesh shape to define static shaped arrays which use the mesh to reconstruct the source. For a rectangular
mesh, the same number of pixels must be used in the y and x directions.

__Edge Zeroing__

By default, all pixels at the edge of the mesh in the source-plane are forced to solutions of zero brightness by 
the linear algebra solver. This prevents unphysical solutions where pixels at the edge of the mesh reconstruct 
bright surface brightnesses, often because they fit residuals from the lens light subtraction.

For a rectangular mesh, the source code computes edge pixels internally using the known
pixels at the edge of the mesh. 
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

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

mass.centre.centre_0 = af.UniformPrior(lower_limit=0.2, upper_limit=0.4)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.4, upper_limit=-0.2)
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

mesh = al.mesh.RectangularAdaptImage(shape=mesh_shape, weight_power=1.0)

regularization = al.reg.Adapt()

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

galaxy_name_image_dict = {
    "('galaxies', 'lens')": dataset.data,
    "('galaxies', 'source')": dataset.data,
}

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

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
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    settings=al.Settings(
        use_border_relocator=True,
        use_positive_only_solver=True,
        use_mixed_precision=True,
    ),
)

"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**, which pairs
the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
from autofit.non_linear.fitness import Fitness
import time

batch_size = 3

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

param_vector = jnp.array(model.physical_values_from_prior_medians)

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
result = fitness._vmap(parameters)
print(result)
print("JAX Time Taken using VMAP:", time.time() - start)
print("JAX Time Taken per Likelihood:", (time.time() - start) / batch_size)

np.testing.assert_allclose(
    np.array(result),
    -12188236.6799005,
    rtol=1e-4,
    err_msg="rectangular: JAX vmap likelihood mismatch",
)

batched_call = jax.jit(jax.vmap(fitness.call))
lowered = batched_call.lower(parameters)
compiled = lowered.compile()
memory_analysis = compiled.memory_analysis()
print(
    f"Memory {(memory_analysis.output_size_in_bytes + memory_analysis.temp_size_in_bytes) / 1024**2:.3} MB"
)


"""
Output an image of the fit, so that we can inspect that it fits the data as expected.
"""
import autolens.plot as aplt
import os

file_path = os.path.join(al.__version__)

instance = model.instance_from_prior_medians()

fit = analysis.fit_from(instance)

print(f"Figure of Merit = {fit.figure_of_merit}")

np.testing.assert_allclose(
    fit.figure_of_merit,
    -12188236.679900503,
    rtol=1e-4,
    err_msg="rectangular: figure_of_merit mismatch",
)


aplt.plot_array(array=fit.model_images_of_planes_list[1], output=aplt.Output(path=file_path, filename=f"{instrument}_source", format="png"))
aplt.subplot_fit_imaging(fit=fit, output=aplt.Output(path=file_path, filename=f"{instrument}_subplot_fit", format="png"))
aplt.subplot_fit_imaging(fit=fit, output=aplt.Output(path=file_path, filename=f"{instrument}_subplot_of_plane_1", format="png"))
aplt.InversionPlotter(inversion=fit.inversion).subplot_of_mapper(mapper_index=0)
