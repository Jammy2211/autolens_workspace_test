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

"""
__Dataset__

Load and plot the galaxy dataset `operated` via .fits files, which we will fit with 
the model.
"""
dataset_name = "no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
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

dataset = dataset.apply_over_sampling(over_sample_size_lp=1)

positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)

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

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a model where:

 - The galaxy's bulge is a parametric `Sersic` bulge [7 parameters]. 
 - The galaxy's point source emission is a parametric operated `Gaussian` centred on the bulge [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=2.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)


# # Lens:
#
bulge = af.Model(al.lp_linear.Sersic)

bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.03, upper_limit=0.03)
bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.03, upper_limit=0.03)

bulge.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.01, upper_limit=0.1)
bulge.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.01, upper_limit=0.1)

# bulge.intensity = af.UniformPrior(lower_limit=1.0, upper_limit=3.0)
bulge.effective_radius = af.UniformPrior(lower_limit=0.4, upper_limit=0.8)
bulge.sersic_index = af.UniformPrior(lower_limit=2.0, upper_limit=4.0)

mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = af.UniformPrior(lower_limit=0.01, upper_limit=0.03)
mass.centre.centre_1 = af.UniformPrior(lower_limit=0.01, upper_limit=0.03)

mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.01, upper_limit=0.1)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.01, upper_limit=0.1)

mass.einstein_radius = af.UniformPrior(lower_limit=1.0, upper_limit=2.2)

shear = af.Model(al.mp.ExternalShear)

shear.gamma_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)
shear.gamma_2 = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

# Source:

bulge = af.Model(al.lp_linear.Sersic)

bulge.centre.centre_0 = af.UniformPrior(lower_limit=0.01, upper_limit=0.03)
bulge.centre.centre_1 = af.UniformPrior(lower_limit=0.01, upper_limit=0.03)

bulge.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.01, upper_limit=0.1)
bulge.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.01, upper_limit=0.1)

# bulge.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)
bulge.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=1.5)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


# # Lens:
#
# total_gaussians = 30
# gaussian_per_basis = 2
#
# # The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
# mask_radius = 3.0
# log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)
#
# # By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.
#
# centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
# centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
#
# bulge_gaussian_list = []
#
# for j in range(gaussian_per_basis):
#     # A list of Gaussian model components whose parameters are customized belows.
#
#     gaussian_list = af.Collection(
#         af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
#     )
#
#     # Iterate over every Gaussian and customize its parameters.
#
#     for i, gaussian in enumerate(gaussian_list):
#         gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
#         gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
#         gaussian.ell_comps = gaussian_list[
#             0
#         ].ell_comps  # All Gaussians have same elliptical components.
#         gaussian.sigma = (
#             10 ** log10_sigma_list[i]
#         )  # All Gaussian sigmas are fixed to values above.
#
#     bulge_gaussian_list += gaussian_list
#
# # The Basis object groups many light profiles together into a single model component.
#
# bulge = af.Model(
#     al.lp_basis.Basis,
#     profile_list=bulge_gaussian_list,
# )
#
# mass = af.Model(al.mp.Isothermal)
#
# mass.centre.centre_0 = af.UniformPrior(lower_limit=0.01, upper_limit=0.03)
# mass.centre.centre_1 = af.UniformPrior(lower_limit=0.01, upper_limit=0.03)
#
# mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.01, upper_limit=0.1)
# mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.01, upper_limit=0.1)
#
# mass.einstein_radius = af.UniformPrior(lower_limit=1.0, upper_limit=2.2)
#
# shear = af.Model(al.mp.ExternalShear)
#
# shear.gamma_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)
# shear.gamma_2 = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)
#
# lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)
#
# # Source:
#
# total_gaussians = 20
# gaussian_per_basis = 1
#
# # By defining the centre here, it creates two free parameters that are assigned to the source Gaussians.
#
# centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
# centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
#
# log10_sigma_list = np.linspace(-2, np.log10(1.0), total_gaussians)
#
# bulge_gaussian_list = []
#
# for j in range(gaussian_per_basis):
#     gaussian_list = af.Collection(
#         af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
#     )
#
#     for i, gaussian in enumerate(gaussian_list):
#         gaussian.centre.centre_0 = centre_0
#         gaussian.centre.centre_1 = centre_1
#         gaussian.ell_comps = gaussian_list[0].ell_comps
#         gaussian.sigma = 10 ** log10_sigma_list[i]
#
#     bulge_gaussian_list += gaussian_list
#
# source_bulge = af.Model(
#     al.lp_basis.Basis,
#     profile_list=bulge_gaussian_list,
# )
#
# source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)
#
# # Overall Lens Model:
#
# model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

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
    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
)


"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**, which pairs
the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
from autofit.non_linear.fitness import Fitness

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

"""
We now test the JAX-ing of this LH function.
"""
parameters = model.physical_values_from_prior_medians
func = jax.jit(fitness)
print(func(parameters))
