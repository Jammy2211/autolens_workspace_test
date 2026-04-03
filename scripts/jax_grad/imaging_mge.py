"""
Tests that jax.value_and_grad can compute finite, non-NaN gradients of the log-likelihood
for an imaging model with parametric light profiles. This tests the core JAX differentiability
that enables gradient-based inference.
"""

import numpy as np
import jax
import jax.numpy as jnp
from os import path

import autofit as af
import autolens as al

dataset_name = "source_complex"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.05,
)

mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

# Lens: NFWSph mass + ExternalShear (no lens light, as the lmp_linear GaussianGradient
# combined light-mass model has unresolved gradient issues under JAX autodiff).

mass = af.Model(al.mp.NFWSph)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source: MGE with lp_linear.Gaussian (same as mge.py, using mask_radius=3.0 inner).

mask_radius_source = 3.0

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius_source, total_gaussians=20, centre_prior_is_uniform=False
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

analysis = al.AnalysisImaging(
    dataset=dataset,
)

from autofit.non_linear.fitness import Fitness

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

param_vector = jnp.array(model.physical_values_from_prior_medians)

# Perturb ell_comps away from (0,0) to avoid degenerate gradients at the
# circular-profile singularity (arctan2 gradient is undefined at exactly (0,0)).
key = jax.random.PRNGKey(0)
perturbation = jax.random.uniform(key, shape=param_vector.shape, minval=0.01, maxval=0.05)
param_vector = param_vector + perturbation

value, grad = jax.value_and_grad(fitness.call)(param_vector)

print(f"Log likelihood = {float(value):.6f}")
print(f"Gradient shape = {grad.shape}")
print(f"Gradient = {np.array(grad)}")

assert np.isfinite(float(value)), "Log likelihood is not finite"
assert grad.shape == (model.total_free_parameters,), f"Gradient shape mismatch: {grad.shape}"
assert np.all(np.isfinite(np.array(grad))), f"Gradient contains non-finite values: {np.array(grad)}"
assert not np.all(np.array(grad) == 0.0), "Gradient is all zeros"

print("imaging_mge.py JAX gradient checks passed.")
