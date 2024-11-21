"""
SLaM (Source, Light and Mass): Source Light Pixelized + Light Profile + Mass Total + Subhalo NFW
================================================================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS PIPELINE.

Using a SOURCE LP PIPELINE, LIGHT LP PIPELINE, MASS PIPELINE and SUBHALO PIPELINE this SLaM script
fits `Imaging` of a strong lens system, where in the final model:

 - The lens galaxy's light is a bulge+disk `Sersic` and `Exponential`.
 - The lens galaxy's total mass distribution is an `Isothermal`.
 - A dark matter subhalo near The lens galaxy mass is included as a`NFWMCRLudlowSph`.
 - The source galaxy is an `Inversion`.

This uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`
 `subhalo/detection`

Check them out for a full description of the analysis!
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
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=mask_radius
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
    path_prefix=path.join("slam", "source_pix", "mass_total", "extra_galaxies"),
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
__Clump Model__ 

This model includes extra_galaxies, which are `Galaxy` objects with light and mass profiles fixed to an input centre which 
model galaxies nearby the strong lens system.

A full description of the extra galaxies API is given in the 
script `autolens_workspace/*/imaging/modeling/features/extra_galaxies.py`
"""
# Extra Galaxies:

extra_galaxies_centres = al.Grid2DIrregular(values=[(1.0, 1.0), [2.0, 2.0]])

extra_galaxies_dict = {}

for i, extra_galaxy_centre in enumerate(extra_galaxies_centres):
    extra_galaxy = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=al.lp_linear.SersicSph(centre=extra_galaxy_centre),
        mass=al.mp.IsothermalSph(centre=extra_galaxy_centre),
    )

    extra_galaxy.mass.einstein_radius = af.UniformPrior(
        lower_limit=0.0, upper_limit=0.1
    )

    extra_galaxies_dict[f"extra_galaxy_{i}"] = extra_galaxy

extra_galaxies = af.Collection(**extra_galaxies_dict)

"""
__SOURCE LP PIPELINE (with lens light)__

The SOURCE LP PIPELINE (with lens light) uses three searches to initialize a robust model for the 
source galaxy's light, which in this example:
 
 - Uses a parametric `Sersic` bulge and `Exponential` disk with centres aligned for the lens
 galaxy's light.
 
 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

 Settings:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=dataset)

# Lens Light

centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

total_gaussians = 30
gaussian_per_basis = 2

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

# Source Light

centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

total_gaussians = 30
gaussian_per_basis = 1

log10_sigma_list = np.linspace(-3, np.log10(1.0), total_gaussians)

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

source_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

# Extra Galaxies:

extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:
    # Extra Galaxy Light

    total_gaussians = 10

    ### FUTURE IMPROVEMENT: Set the size based on each extra galaxy's size as opposed to the mask.

    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    ### FUTURE IMPROVEMENT: Use elliptical Gaussians for the extra galaxies where the ellipticity is estimated beforehand.

    extra_galaxy_gaussian_list = []

    gaussian_list = af.Collection(
        af.Model(al.lp_linear.GaussianSph) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = extra_galaxy_centre[0]
        gaussian.centre.centre_1 = extra_galaxy_centre[1]
        gaussian.sigma = 10 ** log10_sigma_list[i]

    extra_galaxy_gaussian_list += gaussian_list

    extra_galaxy_bulge = af.Model(
        al.lp_basis.Basis, profile_list=extra_galaxy_gaussian_list
    )

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre = extra_galaxy_centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)

    extra_galaxy = af.Model(
        al.Galaxy, redshift=0.5, bulge=extra_galaxy_bulge, mass=mass
    )

    extra_galaxy.mass.centre = extra_galaxy_centre

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

source_lp_result = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=lens_bulge,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
    extra_galaxies=extra_galaxies,
)
"""
__SOURCE PIX PIPELINE__

The SOURCE PIX PIPELINE (and every pipeline that follows) are identical to the `start_here.ipynb` example.

The model components for the extra galaxies (e.g. `lens_bulge` and `lens_disk`) are passed from the SOURCE LP PIPELINE,
via the `source_lp_result` object, therefore you do not need to manually pass them below.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
    positions_likelihood=source_lp_result.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

# Extra Galaxies:

extra_galaxies = source_lp_result.model.extra_galaxies

for galaxy, result_galaxy in zip(extra_galaxies, source_lp_result.instance.extra_galaxies):

    galaxy.bulge = result_galaxy.bulge

source_pix_result_1 = slam.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=al.mesh.Delaunay,
    extra_galaxies=extra_galaxies,
)

"""
__SOURCE PIX PIPELINE 2 (with lens light)__

As above, this pipeline also has the same API as the `start_here.ipynb` example.

The extra galaxies are passed from the SOURCE PIX PIPELINE, via the `source_pix_result_1` object, therefore you do not
need to manually pass them below.
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
    mesh=al.mesh.Delaunay,
    regularization=al.reg.AdaptiveBrightnessSplit,
)


"""
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE PIX PIPELINE.
In this example it:

 - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
 light [Do not use the results of the SOURCE LP PIPELINE to initialize priors].

 - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE PIX PIPELINE].

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values].
"""
centre_0 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
centre_1 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)

total_gaussians = 30
gaussian_per_basis = 2

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):

    gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians))

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

analysis = al.AnalysisImaging(
    dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1)
)


# Extra Galaxies:

extra_galaxies = source_lp_result.model.extra_galaxies

for galaxy, result_galaxy in zip(extra_galaxies, source_pix_result_1.instance.extra_galaxies):

    galaxy.mass = result_galaxy.mass

light_result = slam.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=lens_bulge,
    lens_disk=None,
    extra_galaxies=extra_galaxies,
)

"""
__MASS TOTAL PIPELINE (with lens light)__

The MASS TOTAL PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIPELINE to initialize the model priors and the lens light
model of the LIGHT LP PIPELINE. In this example it:

 - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
 light [fixed from LIGHT LP PIPELINE].

 - Uses an `PowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE 
 PARAMETRIC PIPELINE + centre unfixed from (0.0, 0.0)].
 
 - Uses the `Sersic` model representing a bulge for the source's light [priors initialized from SOURCE 
 PARAMETRIC PIPELINE].
 
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS PIPELINE.
"""
analysis = al.AnalysisImaging(
    dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1)
)

# Extra Galaxies:

extra_galaxies = source_lp_result.model.extra_galaxies

for galaxy, result_galaxy in zip(extra_galaxies, light_result.instance.extra_galaxies):

    galaxy.bulge = result_galaxy.bulge

mass_result = slam.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
    extra_galaxies=extra_galaxies,
)

"""
__SUBHALO PIPELINE (single plane detection)__

The SUBHALO PIPELINE (single plane detection) consists of the following searches:
 
 1) Refit the lens and source model, to refine the model evidence for comparing to the models fitted which include a 
 subhalo. This uses the same model as fitted in the MASS PIPELINE. 
 2) Performs a grid-search of non-linear searches to attempt to detect a dark matter subhalo. 
 3) If there is a successful detection a final search is performed to refine its parameters.
 
For this runner the SUBHALO PIPELINE customizes:

 - The [number_of_steps x number_of_steps] size of the grid-search, as well as the dimensions it spans in arc-seconds.
 - The `number_of_cores` used for the gridsearch, where `number_of_cores > 1` performs the model-fits in paralle using
 the Python multiprocessing module.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
)

# Extra Galaxies:

extra_galaxies = mass_result.model.extra_galaxies

for galaxy, result_galaxy in zip(extra_galaxies, light_result.instance.extra_galaxies):

    galaxy.bulge = result_galaxy.bulge

subhalo_result_1 = slam.subhalo.detection.run_1_no_subhalo(
    settings_search=settings_search,
    analysis=analysis,
    mass_result=mass_result,
    extra_galaxies=extra_galaxies,
)

subhalo_grid_search_result_2 = slam.subhalo.detection.run_2_grid_search(
    settings_search=settings_search,
    analysis=analysis,
    mass_result=mass_result,
    subhalo_result_1=subhalo_result_1,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
    extra_galaxies=extra_galaxies,
)

subhalo_result_3 = slam.subhalo.detection.run_3_subhalo(
    settings_search=settings_search,
    analysis=analysis,
    subhalo_result_1=subhalo_result_1,
    subhalo_grid_search_result_2=subhalo_grid_search_result_2,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    extra_galaxies=extra_galaxies,
)

"""
Finish.
"""
