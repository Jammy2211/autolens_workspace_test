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

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "slam"))

import autofit as af
import autolens as al
import autolens.plot as aplt
import slam

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

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=path.join(
        "slam",
        "source_pix",
        "mass_light_dark",
        "basis",
    ),
    number_of_cores=1,
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

 - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens
 galaxy's light.

 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

 Settings:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS LIGHT DARK 
 PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=dataset)

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

total_gaussians = 3
gaussian_per_basis = 1

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]

    bulge_gaussian_list += gaussian_list

lens_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

disk_gaussian_list = []

for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]

    disk_gaussian_list += gaussian_list

lens_disk = af.Model(
    al.lp_basis.Basis,
    profile_list=disk_gaussian_list,
)

source_lp_result = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=lens_bulge,
    lens_disk=lens_disk,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.Sersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__SOURCE PIX PIPELINE (with lens light)__

The SOURCE PIX PIPELINE (with lens light) uses two searches to initialize a robust model for the pixelization
that reconstructs the source galaxy's light. It begins by fitting a `Voronoi` pixelization with `Constant` 
regularization, to set up the model and hyper images, and then:

 - Uses a `Voronoi` pixelization.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
 SOURCE PIX PIPELINE.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
)

source_pix_result_1 = slam.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=al.mesh.Voronoi,
)

"""
__SOURCE PIX PIPELINE 2 (with lens light)__
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    settings_inversion=al.SettingsInversion(
        image_mesh_min_mesh_pixels_per_pixel=3,
        image_mesh_min_mesh_number=5,
        image_mesh_adapt_background_percent_threshold=0.1,
        image_mesh_adapt_background_percent_check=0.8,
    ),
)

source_pix_result_2 = slam.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    image_mesh=al.image_mesh.Hilbert,
    mesh=al.mesh.Voronoi,
    regularization=al.reg.AdaptiveBrightnessSplit,
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
centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

total_gaussians = 3
gaussian_per_basis = 1

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]

    bulge_gaussian_list += gaussian_list

lens_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

disk_gaussian_list = []

for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]

    disk_gaussian_list += gaussian_list

lens_disk = af.Model(
    al.lp_basis.Basis,
    profile_list=disk_gaussian_list,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
)

light_result = slam.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=lens_bulge,
    lens_disk=lens_disk,
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
    dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1)
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
    light_result=light_result,
    settings_search=settings_search
)


dark = af.Model(al.mp.NFWMCRLudlow)

mass_result = slam.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    dark=dark,
    link_mass_to_light_ratios=True
)

mass_result = slam.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    use_gradient=True,
    dark=dark,
    link_mass_to_light_ratios=True
)

dark = af.Model(al.mp.NFWMCRLudlowSph)

mass_result = slam.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    dark=dark,
    link_mass_to_light_ratios=True
)

mass_result = slam.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    use_gradient=True,
    dark=dark,
    link_mass_to_light_ratios=True
)


dark = af.Model(al.mp.NFWMCRLudlowSph)

dark = None

mass_result = slam.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=source_pix_result_2,
    use_gradient=True,
    dark=dark,
)

mass_result = slam.mass_light_dark.run(
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

mass_result = slam.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=source_pix_result_2,
    use_gradient=True,
    dark=dark,
)

mass_result = slam.mass_light_dark.run(
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
