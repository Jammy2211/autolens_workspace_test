def fit():
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

    import os
    import numpy as np
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
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=mask_radius
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

    # dataset = dataset.apply_w_tilde()

    """
    __Settings AutoFit__
    
    The settings of autofit, which controls the output paths, parallelization, database use, etc.
    """
    settings_search = af.SettingsSearch(
        path_prefix=Path("slam") / "source_pix" / "mass_total" / "base",
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
    image_mesh = al.image_mesh.Overlay(shape=(26, 26))

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
    lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE LP PIPELINE.
    In this example it:

     - Uses a multi Gaussian expansion with 2 sets of 30 Gaussians for the lens galaxy's light. [6 Free Parameters].

     - Uses an `Isothermal` mass model with `ExternalShear` for the lens's total mass distribution [fixed from SOURCE PIX PIPELINE].

     - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

     - Carries the lens redshift and source redshift of the SOURCE PIPELINE through to the MASS PIPELINE [fixed values].   
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
    __MASS TOTAL PIPELINE (with lens light)__

    The MASS TOTAL PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
    using the lens mass model and source model of the SOURCE PIX PIPELINE to initialize the model priors and the lens 
    light model of the LIGHT LP PIPELINE. In this example it:

     - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
     light [fixed from LIGHT LP PIPELINE].

     - Uses an `PowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE 
     PARAMETRIC PIPELINE + centre unfixed from (0.0, 0.0)].

     - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIX PIPELINE].

     - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS TOTAL 
     PIPELINE.

    __Settings__:

     - adapt: We may be using hyper features and therefore pass the result of the SOURCE PIX PIPELINE to use as the
     hyper dataset if required.

     - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
     in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
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

    mass_result = slam_pipeline.mass_total.run(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        mass=af.Model(al.mp.PowerLaw),
        reset_shear_prior=True,
    )

    """
    __SUBHALO PIPELINE (single plane detection)__

    The SUBHALO PIPELINE (single plane detection) consists of the following searches:

     1) Refit the lens and source model, to refine the model evidence for comparing to the models fitted which include a 
     subhalo. This uses the same model as fitted in the MASS TOTAL PIPELINE. 
     2) Performs a grid-search of non-linear searches to attempt to detect a dark matter subhalo. 
     3) If there is a successful detection a final search is performed to refine its parameters.

    For this runner the SUBHALO PIPELINE customizes:

     - The [number_of_steps x number_of_steps] size of the grid-search, as well as the dimensions it spans in arc-seconds.
     - The `number_of_cores` used for the gridsearch, where `number_of_cores > 1` performs the model-fits in paralle using
     the Python multiprocessing module.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood=mass_result.positions_likelihood_from(
            factor=2.0, minimum_threshold=0.2,
        ),
        adapt_images=adapt_images,
        preloads=preloads,
        use_jax=True,
    )

    from autogalaxy.cosmology.wrap import Planck15

    # subhalo_mass = af.Model(al.mp.gNFWVirialMassConcSph)

    # # properties to fix
    # subhalo_mass.cosmology = Planck15()
    # subhalo_mass.overdens = 200
    # subhalo_mass.redshift_object = 0.2
    # subhalo_mass.redshift_source = 1.0
    #
    # # set priors
    # # uses default Uniform priors [7,12] for log10m_vir
    # subhalo_mass.centre_0 = af.GaussianPrior(mean=1.0, sigma=1.0)
    # subhalo_mass.centre_1 = af.GaussianPrior(mean=1.0, sigma=1.0)

    subhalo_result_1 = slam_pipeline.subhalo.detection.run_1_no_subhalo(
        settings_search=settings_search,
        analysis=analysis,
        mass_result=mass_result,
    )

    subhalo_grid_search_result_2 = slam_pipeline.subhalo.detection.run_2_grid_search(
        settings_search=settings_search,
        analysis=analysis,
        mass_result=mass_result,
        subhalo_result_1=subhalo_result_1,
        subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
        grid_dimension_arcsec=3.0,
        number_of_steps=2
    )

    subhalo_result_3 = slam_pipeline.subhalo.detection.run_3_subhalo(
        settings_search=settings_search,
        analysis=analysis,
        subhalo_result_1=subhalo_result_1,
        subhalo_grid_search_result_2=subhalo_grid_search_result_2,
        subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    )


if __name__ == "__main__":

    fit()
