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
    from os import path

    import autofit as af
    import autolens as al
    import autolens.plot as aplt
    import slam_pipeline

    """
    __Dataset__ 
    
    Load the `Imaging` data, define the `Mask2D` and plot them.
    """
    real_space_mask = al.Mask2D.circular(
        shape_native=(200, 200), pixel_scales=0.1, radius=2.0
    )

    dataset_name = "sma"
    dataset_path = path.join("dataset", "interferometer", "instruments", dataset_name)

    dataset = al.Interferometer.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerNUFFT,
    )

    """
    __Settings AutoFit__
    
    The settings of autofit, which controls the output paths, parallelization, database use, etc.
    """
    settings_search = af.SettingsSearch(
        path_prefix=path.join("slam", "source_pix", "mass_total", "interferometer"),
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
     
     - Uses a parametric `Sersic` bulge and `Exponential` disk with centres aligned for the lens
     galaxy's light.
     
     - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.
    
     Settings:
    
     - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS PIPELINE).
    """
    analysis = al.AnalysisImaging(dataset=dataset)

    bulge = af.Model(al.lp.Sersic)
    disk = af.Model(al.lp.Exponential)
    bulge.centre = disk.centre

    source_lp_result = slam.source_lp.run(
        settings_search=settings_search,
        analysis=analysis,
        lens_bulge=bulge,
        lens_disk=disk,
        mass=af.Model(al.mp.Isothermal),
        shear=af.Model(al.mp.ExternalShear),
        source_bulge=af.Model(al.lp.Sersic),
        mass_centre=(0.0, 0.0),
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
    )

    """
    __SOURCE PIX PIPELINE 1 (with lens light)__
    
    The SOURCE PIX PIPELINE (with lens light) uses two searches to initialize a robust model for the pixelization
    that reconstructs the source galaxy's light. 
    
    This pixelization adapts its source pixels to the morphology of the source, placing more pixels in its 
    brightest regions. To do this, an "adapt image" is required, which is the lens light subtracted image meaning
    only the lensed source emission is present.
    
    The SOURCE LP Pipeline result is not good enough quality to set up this adapt image (e.g. the source
    may be more complex than a light profile). The first step of the SOURCE PIX PIPELINE therefore fits a new
    model using a pixelization to create this adapt image.
        
    It fits a `Voronoi` pixelization, `Constant` regularization, to set up the model and hyper images, and then:
    
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
    
     - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE PIX PIPELINE].
    
     - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIX PIPELINE].
    
     - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
     PIPELINE [fixed values].
    """
    bulge = af.Model(al.lp.Sersic)
    disk = af.Model(al.lp.Exponential)
    bulge.centre = disk.centre

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    )

    light_result = slam.light_lp.run(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        lens_bulge=bulge,
        lens_disk=disk,
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
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    )

    mass_result = slam.mass_total.run(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        mass=af.Model(al.mp.PowerLaw),
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

    subhalo_result_1 = slam.subhalo.detection.run_1_no_subhalo(
        settings_search=settings_search,
        analysis=analysis,
        mass_result=mass_result,
    )

    subhalo_grid_search_result_2 = slam.subhalo.detection.run_2_grid_search(
        settings_search=settings_search,
        analysis=analysis,
        mass_result=mass_result,
        subhalo_result_1=subhalo_result_1,
        subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
        grid_dimension_arcsec=3.0,
        number_of_steps=2,
    )

    subhalo_result_3 = slam.subhalo.detection.run_3_subhalo(
        settings_search=settings_search,
        analysis=analysis,
        subhalo_result_1=subhalo_result_1,
        subhalo_grid_search_result_2=subhalo_grid_search_result_2,
        subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    )

    """
    Finish.
    """


if __name__ == "__main__":

    fit()
