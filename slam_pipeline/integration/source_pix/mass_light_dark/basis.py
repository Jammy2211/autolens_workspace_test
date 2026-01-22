"""
SLaM (Source, Light and Mass): Light Parametric + Mass Total + Source Parametric
================================================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS LIGHT DARK PIPELINE.

Using a SOURCE LP PIPELINE, LIGHT PIPELINE and a MASS LIGHT DARK PIPELINE this SLaM script fits `Imaging` of
a strong lens system, where in the final model:

 - The lens galaxy's light is a bulge+disk `Sersic` and `Sersic`.
 - The lens galaxy's stellar mass distribution is a bulge+disk tied to the light model above.
 - The lens galaxy's dark matter mass distribution is modeled as a `NFWMCRLudlow`.
 - The source galaxy's light is a parametric `Inversion`.

This runner uses the SLaM pipelines:

 `source_lp`
 `source_pix/with_lens_light`
 `light_lp`
 `mass_total/mass_light_dark`

Check them out for a detailed description of the analysis!
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import os
from os import path
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt
import slam_pipeline

"""
__Dataset__ 

Load the `Imaging` data, define the `Mask2D` and plot them.
"""
dataset_name = "with_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.2,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.1, 0.3],
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("slam") / "source_pix" / "mass_light_dark" / "base",
    session=None,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).
"""
redshift_lens = 0.5
redshift_source = 1.0


"""
__SOURCE LP PIPELINE (with lens light)__

The SOURCE LP PIPELINE (with lens light) uses three searches to initialize a robust model for the 
source galaxy's light, which in this example:

 - Uses a parametric `Sersic` bulge and `Exponential` disk with centres aligned for the lens
 galaxy's light.

 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

 __Settings__:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
)

# Lens Light

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=30,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

mass = af.Model(al.mp.Isothermal)

# Source:

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)

source_lp_result = slam_pipeline.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=lens_bulge,
    lens_disk=None,
    mass=mass,
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

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
"""
image_mesh = al.image_mesh.Overlay(shape=(10, 10))

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask,
)

image_plane_mesh_grid_edge_pixels = 30

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=image_plane_mesh_grid_edge_pixels,
)

total_mapper_pixels = image_plane_mesh_grid.shape[0]

total_linear_light_profiles = 60

mapper_indices = al.mapper_indices_from(
    total_linear_light_profiles=total_linear_light_profiles,
    total_mapper_pixels=total_mapper_pixels,
)

# Extract the last `image_plane_mesh_grid_edge_pixels` indices, which correspond to the circle edge points we added

source_pixel_zeroed_indices = mapper_indices[
                              -image_plane_mesh_grid_edge_pixels:
                              ]

preloads = al.Preloads(
    mapper_indices=mapper_indices,
    source_pixel_zeroed_indices=source_pixel_zeroed_indices
)

"""
__SOURCE PIX PIPELINE (with lens light)__

The SOURCE PIX PIPELINE (with lens light) uses four searches to initialize a robust model for the `Inversion` 
that reconstructs the source galaxy's light. It begins by fitting a `VoronoiMagnification` pixelization with `Constant` 
regularization, to set up the model and hyper images, and then:

 - Uses a `VoronoiBrightnessImage` pixelization.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
 SOURCE PIX PIPELINE.
"""
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_lp_result
)

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

signal_to_noise_threshold = 3.0
over_sample_size_pixelization = np.where(galaxy_image_name_dict["('galaxies', 'source')"] > signal_to_noise_threshold, 4, 2)
over_sample_size_pixelization = al.Array2D(values=over_sample_size_pixelization, mask=mask)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=over_sample_size_pixelization
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[source_lp_result.positions_likelihood_from(
        factor=2.0, minimum_threshold=0.3,
    )],
    preloads=preloads,
    use_jax=True,
)

source_pix_result_1 = slam_pipeline.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=al.mesh.Delaunay(),
    regularization_init=af.Model(al.reg.AdaptiveBrightnessSplit),
)


galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=source_pix_result_1)

image_mesh = al.image_mesh.Hilbert(pixels=1000, weight_power=1.0, weight_floor=0.0001)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
)

image_plane_mesh_grid_edge_pixels = 30

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=image_plane_mesh_grid_edge_pixels,
)

total_mapper_pixels = image_plane_mesh_grid.shape[0]

mapper_indices = al.mapper_indices_from(
    total_linear_light_profiles=total_linear_light_profiles,
    total_mapper_pixels=total_mapper_pixels,
)

source_pixel_zeroed_indices = mapper_indices[
                              -image_plane_mesh_grid_edge_pixels:
                              ]

preloads = al.Preloads(
    mapper_indices=mapper_indices,
    source_pixel_zeroed_indices=source_pixel_zeroed_indices
)

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)


analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
    use_jax=True,
)

source_pix_result_2 = slam_pipeline.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=al.mesh.Delaunay(),
    regularization=af.Model(al.reg.AdaptiveBrightnessSplit),
)
"""
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE PIX PIPELINE.
In this example it:

 - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
 light [Do not use the results of the SOURCE LP PIPELINE to initialize priors].

 - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE LP PIPELINE].

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values].
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
    use_jax=True,
)

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=30,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True
)

light_result = slam_pipeline.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=lens_bulge,
    lens_disk=None,
)

"""
__MASS LIGHT DARK PIPELINE (with lens light)__

The MASS LIGHT DARK PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of 
accuracy, using the source model of the SOURCE PIPELINE and the lens light model of the LIGHT LP PIPELINE to 
initialize the model priors . In this example it:

 - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
 light and its stellar mass [12 parameters: fixed from LIGHT LP PIPELINE].

 - The lens galaxy's dark matter mass distribution is a `NFWMCRLudlow` whose centre is aligned with bulge of 
 the light and stellar mass model above [5 parameters].

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the MASS 
 LIGHT DARK PIPELINE.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        source_pix_result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
    preloads=preloads,
    use_jax=True,
)

"""
When linear light profiles are used in the LIGHT PIPELINE, their intensities are solved for via linear algebra
when setting up their corresponding light and mass profiles for the MASS LIGHT DARK PIPELINE.

This calculation is not numerically accuracy to a small amount (of order 1e-8) in the `intensity` values that
are solved for. This lack of accuracy will not impact the lens modeling in a noticeable way.

However, it does mean that when a pipeine is rerun the `intensity` values that are solved for may change by a small
amount, changing the unique identifier created for the fit where results are stored, meaning a run does not
resume correctly.

The code below therefore outputs the chaining tracer used to pass the `intensity` values to a .json file, or loads
it from this file if it is already there. This ensures that when a pipeline is rerun, the same `intensity`
values are always used.
"""
lp_chain_tracer = al.util.chaining.lp_chain_tracer_from(
    light_result=light_result, settings_search=settings_search
)


dark = af.Model(al.mp.NFWMCRLudlow)

mass_result = slam_pipeline.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    dark=dark,
    link_mass_to_light_ratios=True,
)

mass_result = slam_pipeline.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    use_gradient=True,
    dark=dark,
    link_mass_to_light_ratios=True,
)

dark = af.Model(al.mp.NFWMCRLudlowSph)

mass_result = slam_pipeline.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    dark=dark,
    link_mass_to_light_ratios=True,
)

mass_result = slam_pipeline.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    use_gradient=True,
    dark=dark,
    link_mass_to_light_ratios=True,
)


dark = af.Model(al.mp.NFWMCRLudlowSph)

dark = None

mass_result = slam_pipeline.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=source_pix_result_2,
    use_gradient=True,
    dark=dark,
)

mass_result = slam_pipeline.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=source_pix_result_2,
    use_gradient=False,
    dark=dark,
)

dark = af.Model(al.mp.NFWMCRLudlow)

dark = None

mass_result = slam_pipeline.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=source_pix_result_2,
    use_gradient=True,
    dark=dark,
)

mass_result = slam_pipeline.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=source_pix_result_2,
    use_gradient=False,
    dark=dark,
)


"""
Finish.
"""
