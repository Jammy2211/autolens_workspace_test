# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from jaxns.src.jaxns.framework.prior import Prior
from jaxns.src.jaxns.framework.model import Model
from jaxns.src.jaxns.public import NestedSampler
from jax import random
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad

from os import path

import autofit as af
import autolens as al
from autoconf import conf

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
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=mask_radius
)

dataset = dataset.apply_mask(mask=mask)

dataset = dataset.apply_over_sampling(over_sample_size_lp=1)

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

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

# Source:

bulge = af.Model(al.lp_linear.Sersic)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
can compute its gradient.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    #   positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
)

# search = af.Nautilus(
#     name="imaging_lp_vectorized_2",
#     unique_tag=dataset_name,
#     n_live=150,
#     vectorized=True,
#     iterations_per_full_update=1000,
# )
#
# result = search.fit(model=model, analysis=analysis)


import tensorflow_probability.substrates.jax as tfp

tfpd = tfp.distributions


def prior_transform():

    lens_bulge_centre_0 = yield Prior(
        tfpd.Uniform(-0.3, 0.3), name="lens_bulge_centre_0"
    )
    lens_bulge_centre_1 = yield Prior(
        tfpd.Uniform(-0.3, 0.3), name="lens_bulge_centre_1"
    )
    lens_bulge_ell_comps_0 = yield Prior(
        tfpd.Uniform(-0.5, 0.5), name="lens_bulge_ell_comps_0"
    )
    lens_bulge_ell_comps_1 = yield Prior(
        tfpd.Uniform(-0.5, 0.5), name="lens_bulge_ell_comps_1"
    )
    lens_bulge_effective_radius = yield Prior(
        tfpd.Uniform(0.0, 30.0), name="lens_bulge_effective_radius"
    )
    lens_bulge_sersic_index = yield Prior(
        tfpd.Uniform(0.5, 5.0), name="lens_bulge_sersic_index"
    )

    lens_mass_centre_0 = yield Prior(tfpd.Uniform(-0.1, 0.1), name="lens_mass_centre_0")
    lens_mass_centre_1 = yield Prior(tfpd.Uniform(-0.1, 0.1), name="lens_mass_centre_1")
    lens_mass_ell_comps_0 = yield Prior(
        tfpd.Uniform(-0.5, 0.5), name="lens_mass_ell_comps_0"
    )
    lens_mass_ell_comps_1 = yield Prior(
        tfpd.Uniform(-0.5, 0.5), name="lens_mass_ell_comps_1"
    )
    lens_mass_einstein_radius = yield Prior(
        tfpd.Uniform(0.0, 8.0), name="lens_mass_einstein_radius"
    )

    lens_shear_gamma_1 = yield Prior(tfpd.Uniform(-0.2, 0.2), name="lens_shear_gamma_1")
    lens_shear_gamma_2 = yield Prior(tfpd.Uniform(-0.2, 0.2), name="lens_shear_gamma_2")

    source_bulge_centre_0 = yield Prior(
        tfpd.Uniform(-2.0, 2.0), name="source_bulge_centre_0"
    )
    source_bulge_centre_1 = yield Prior(
        tfpd.Uniform(-2.0, 2.0), name="source_bulge_centre_1"
    )
    source_bulge_ell_comps_0 = yield Prior(
        tfpd.Uniform(-0.5, 0.5), name="source_bulge_ell_comps_0"
    )
    source_bulge_ell_comps_1 = yield Prior(
        tfpd.Uniform(-0.5, 0.5), name="source_bulge_ell_comps_1"
    )
    source_bulge_effective_radius = yield Prior(
        tfpd.Uniform(0.0, 3.0), name="source_bulge_effective_radius"
    )
    source_bulge_sersic_index = yield Prior(
        tfpd.Uniform(0.5, 5.0), name="source_bulge_sersic_index"
    )

    return (
        lens_bulge_centre_0,
        lens_bulge_centre_1,
        lens_bulge_ell_comps_0,
        lens_bulge_ell_comps_1,
        lens_bulge_effective_radius,
        lens_bulge_sersic_index,
        lens_mass_centre_0,
        lens_mass_centre_1,
        lens_mass_ell_comps_0,
        lens_mass_ell_comps_1,
        lens_mass_einstein_radius,
        lens_shear_gamma_1,
        lens_shear_gamma_2,
        source_bulge_centre_0,
        source_bulge_centre_1,
        source_bulge_ell_comps_0,
        source_bulge_ell_comps_1,
        source_bulge_effective_radius,
        source_bulge_sersic_index,
    )


from autofit.non_linear.fitness import Fitness

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)


def log_likelihood(
    lens_bulge_centre_0,
    lens_bulge_centre_1,
    lens_bulge_ell_comps_0,
    lens_bulge_ell_comps_1,
    lens_bulge_effective_radius,
    lens_bulge_sersic_index,
    lens_mass_centre_0,
    lens_mass_centre_1,
    lens_mass_ell_comps_0,
    lens_mass_ell_comps_1,
    lens_mass_einstein_radius,
    lens_shear_gamma_1,
    lens_shear_gamma_2,
    source_bulge_centre_0,
    source_bulge_centre_1,
    source_bulge_ell_comps_0,
    source_bulge_ell_comps_1,
    source_bulge_effective_radius,
    source_bulge_sersic_index,
):

    params = [
        lens_bulge_centre_0,
        lens_bulge_centre_1,
        lens_bulge_ell_comps_0,
        lens_bulge_ell_comps_1,
        lens_bulge_effective_radius,
        lens_bulge_sersic_index,
        lens_mass_centre_0,
        lens_mass_centre_1,
        lens_mass_ell_comps_0,
        lens_mass_ell_comps_1,
        lens_mass_einstein_radius,
        lens_shear_gamma_1,
        lens_shear_gamma_2,
        source_bulge_centre_0,
        source_bulge_centre_1,
        source_bulge_ell_comps_0,
        source_bulge_ell_comps_1,
        source_bulge_effective_radius,
        source_bulge_sersic_index,
    ]

    return fitness.__call__(params)


model_ns = Model(prior_model=prior_transform, log_likelihood=log_likelihood)

ns = NestedSampler(
    model=model_ns, s=10, k=model_ns.U_ndims, num_live_points=model_ns.U_ndims * 1000
)

termination_reason, state = jax.jit(ns)(random.PRNGKey(432345987))
results = ns.to_results(termination_reason=termination_reason, state=state)
