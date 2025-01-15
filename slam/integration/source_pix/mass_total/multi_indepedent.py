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
 - The source galaxy is an `Sersic`.

This uses the SLaM pipelines:

 `source_lp`
 `light_lp`
 `mass_total`
 `subhalo/detection`

Check them out for a full description of the analysis!
"""


def fit():
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
    dataset_name = "lens_sersic"
    dataset_main_path = path.join("dataset", "multi", dataset_name)

    dataset_waveband = "g"

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_main_path, f"{dataset_waveband}_data.fits"),
        noise_map_path=path.join(dataset_main_path, f"{dataset_waveband}_noise_map.fits"),
        psf_path=path.join(dataset_main_path, f"{dataset_waveband}_psf.fits"),
        pixel_scales=0.08,
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
        path_prefix=path.join("slam", "source_pix", "mass_total", "multi_independent"),
        unique_tag=f"{dataset_name}_data_{dataset_waveband}",
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

    centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    total_gaussians = 30
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
        regularization=al.reg.ConstantZeroth(
            coefficient_neighbor=0.0, coefficient_zeroth=1.0
        ),
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
        regularization=al.reg.ConstantZeroth(
            coefficient_neighbor=0.0, coefficient_zeroth=1.0
        ),
    )

    centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

    total_gaussians = 30
    gaussian_per_basis = 1

    log10_sigma_list = np.linspace(-2, np.log10(1.0), total_gaussians)

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
        regularization=al.reg.ConstantZeroth(
            coefficient_neighbor=0.0, coefficient_zeroth=1.0
        ),
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
    lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE LP PIPELINE.
    In this example it:

     - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
     light [Do not use the results of the SOURCE LP PIPELINE to initialize priors].

     - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE LP PIPELINE].

     - Uses the `Sersic` model representing a bulge for the source's light [fixed from SOURCE LP PIPELINE].

     - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
     PIPELINE [fixed values].
    """
    centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    total_gaussians = 30
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
        dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_lp_result)
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
    __Second Dataset Fits__
    
    We now fit the secondary multi-wavelength datasets, which are lower resolution than the main dataset. 
    
    This uses a for loop to iterate over every waveband of every dataset, load and mask the data and fit it.
    
    Each fit uses a fixed mass model, the lens and source light models update via linear algebra and offsets are
    includded (see full description above).
    
    Its the usual API to set up dataset paths, but include its "main` path which is before the waveband folders.
    """
    dataset_name = "lens_sersic"
    dataset_main_path = path.join("dataset", "multi", dataset_name)

    """
    __Dataset Wavebands__
    
    The following list gives the names of the wavebands we are going to fit. 
    
    The data for each waveband is loaded from a folder in the dataset folder with that name.
    """
    dataset_waveband_list = ["r"]
    pixel_scale_list = [0.12]

    """
    __Dataset Model__
    
    For each fit, the (y,x) offset of the secondary data from the primary data is a free parameter. 
    
    This is achieved by setting up a `DatasetModel` for each waveband, which extends the model with components
    including the grid offset.
    
    This ensures that if the datasets are offset with respect to one another, the model can correct for this,
    with sub-pixel offsets often being important in lens modeling as the precision of a lens model can often be
    less than the requirements on astrometry.
    """
    dataset_model = af.Model(al.DatasetModel)

    dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
        lower_limit=-0.2, upper_limit=0.2
    )
    dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
        lower_limit=-0.2, upper_limit=0.2
    )

    """
    __Result Dict__
    
    Visualization at the end of the pipeline will output all fits to all wavebands on a single matplotlib subplot.
    
    The results of each fit are stored in a dictionary, which is used to pass the results of each fit to the
    visualization functions.
    """
    multi_result_dict = {"g": mass_result}

    for dataset_waveband, pixel_scale in zip(dataset_waveband_list, pixel_scale_list):
        dataset_path = dataset_main_path

        dataset = al.Imaging.from_fits(
            data_path=path.join(dataset_main_path, f"{dataset_waveband}_data.fits"),
            noise_map_path=path.join(
                dataset_main_path, f"{dataset_waveband}_noise_map.fits"
            ),
            psf_path=path.join(dataset_main_path, f"{dataset_waveband}_psf.fits"),
            pixel_scales=pixel_scale,
        )

        mask = al.Mask2D.circular(
            shape_native=dataset.shape_native,
            pixel_scales=dataset.pixel_scales,
            radius=mask_radius,
        )

        dataset = dataset.apply_mask(mask=mask)

        dataset = dataset.apply_over_sampling(
            over_sampling=al.OverSamplingDataset(
                lp=al.OverSampling.over_sample_size_via_radial_bins_from(
                    grid=dataset.grid,
                    sub_size_list=[4, 2, 1],
                    radial_list=[0.1, 0.3],
                    centre_list=[(0.0, 0.0)],
                )
            )
        )

        dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
        dataset_plotter.subplot_dataset()

        """
        __Settings AutoFit__
        """
        settings_search = af.SettingsSearch(
            path_prefix=path.join("slam", "multi", "independent"),
            unique_tag=f"{dataset_name}_data_{dataset_waveband}",
            info=None,
        )

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
        )

        centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
        centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

        total_gaussians = 20
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

        source_lp_result = slam.source_lp.run(
            settings_search=settings_search,
            analysis=analysis,
            lens_bulge=light_result.instance.galaxies.lens.bulge,
            lens_disk=None,
            lens_point=light_result.instance.galaxies.lens.point,
            mass=mass_result.instance.galaxies.lens.mass,
            shear=mass_result.instance.galaxies.lens.shear,
            source_bulge=source_bulge,
            redshift_lens=0.5,
            redshift_source=1.0,
            dataset_model=dataset_model,
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
        analysis = al.AnalysisImaging(
            dataset=dataset,
            adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
            raise_inversion_positions_likelihood_exception=False,
        )

        source_pix_result_1 = slam.source_pix.run_1(
            settings_search=settings_search,
            analysis=analysis,
            source_lp_result=source_lp_result,
            mesh_init=al.mesh.Delaunay,
            dataset_model=dataset_model,
            fixed_mass_model=True
        )

        source_pix_result_1.max_log_likelihood_fit.inversion.cls_list_from(
            cls=al.AbstractMapper
        )[0].extent_from()

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

        multi_result = slam.source_pix.run_2(
            settings_search=settings_search,
            analysis=analysis,
            source_lp_result=source_lp_result,
            source_pix_result_1=source_pix_result_1,
            image_mesh=al.image_mesh.Hilbert,
            mesh=al.mesh.Delaunay,
            regularization=al.reg.AdaptiveBrightnessSplit,
            dataset_model=dataset_model,
        )

        multi_result_dict[dataset_waveband] = multi_result

        slam.slam_util.output_result_to_fits(
            output_path=path.join(dataset_path, "result"),
            result=multi_result,
            model_lens_light=True,
            model_source_light=True,
            source_reconstruction=True,
        )

        slam.slam_util.output_model_results(
            output_path=path.join(dataset_path, "result"),
            result=multi_result,
            filename="sie_model.results",
        )

    tag_list = list(multi_result_dict.keys())

    slam.slam_util.output_fit_multi_png(
        output_path=path.join(dataset_main_path),
        result_list=[multi_result_dict[dataset_waveband] for dataset_waveband in tag_list],
        tag_list=tag_list,
        filename="8_sie_fit",
        main_dataset_index=0,
    )

    slam.slam_util.output_source_multi_png(
        output_path=path.join(dataset_main_path),
        result_list=[multi_result_dict[dataset_waveband] for dataset_waveband in tag_list],
        tag_list=tag_list,
        filename="9_source_reconstruction",
        main_dataset_index=0,
    )



if __name__ == "__main__":
    fit()
