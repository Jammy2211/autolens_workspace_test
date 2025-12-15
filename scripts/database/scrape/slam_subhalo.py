"""
Database: Model-Fit
===================

This is a simple example of a model-fit which we wish to write to the database. This should simply output the
results to the `.sqlite` database file.
"""


def fit():
    # %matplotlib inline
    # from pyprojroot import here
    # workspace_path = str(here())
    # %cd $workspace_path
    # print(f"Working Directory has been set to `{workspace_path}`")

    import os
    from os import path

    os.environ["PYAUTOFIT_TEST_MODE"] = "1"

    import autofit as af
    import autolens as al
    import autolens.plot as aplt
    import slam_pipeline

    """
    __Dataset + Masking__
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

    """
    __Settings AutoFit__
    
    The settings of autofit, which controls the output paths, parallelization, database use, etc.
    """
    settings_search = af.SettingsSearch(
        path_prefix=path.join("database", "scrape", "slam_subhalo"),
        unique_tag=dataset_name,
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

    bulge = af.Model(al.lp_linear.Sersic)
    disk = af.Model(al.lp_linear.Exponential)
    # disk = af.Model(al.lp_linear.Sersic)
    bulge.centre = disk.centre

    source_lp_result = slam.source_lp.run(
        settings_search=settings_search,
        analysis=analysis,
        lens_bulge=bulge,
        lens_disk=disk,
        mass=af.Model(al.mp.Isothermal),
        shear=af.Model(al.mp.ExternalShear),
        source_bulge=af.Model(al.lp_linear.Sersic),
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
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
    bulge = af.Model(al.lp_linear.Sersic)
    disk = af.Model(al.lp_linear.Exponential)
    bulge.centre = disk.centre

    light_result = slam.light_lp.run(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_lp_result,
        source_result_for_source=source_lp_result,
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
        dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_lp_result)
    )

    mass_result = slam.mass_total.run(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_lp_result,
        source_result_for_source=source_lp_result,
        light_result=light_result,
        mass=af.Model(al.mp.PowerLaw),
        multipole_4=af.Model(al.mp.PowerLawMultipole),
        reset_shear_prior=True,
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
        dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_lp_result)
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
    ___Database__
    
    The name of the database, which corresponds to the output results folder.
    """
    database_file = "database_directory_slam_subhalo.sqlite"

    """
    Remove database is making a new build (you could delete manually via your mouse). Building the database is slow, so 
    only do this when you redownload results. Things are fast working from an already built database.
    """
    try:
        os.remove(path.join("output", database_file))
    except FileNotFoundError:
        pass

    """
    Load the database. If the file `slacs.sqlite` does not exist, it will be made by the method below, so its fine if
    you run the code below before the file exists.
    """
    agg = af.Aggregator.from_database(filename=database_file, completed_only=False)

    """
    Add all results in the directory "output/slacs" to the database, which we manipulate below via the agg.
    Avoid rerunning this once the file `slacs.sqlite` has been built.
    """
    agg.add_directory(
        directory=path.join("output", "database", "scrape", "slam_subhalo")
    )

    print("\n\n***********************")
    print("**GRID RESULTS TESTING**")
    print("***********************\n\n")

    print(
        "\n****Total aggregator via `grid_searches` query = ",
        len(agg.grid_searches()),
        "****\n",
    )
    unique_tag = agg.grid_searches().search.unique_tag

    """
    The `GridSearchResult` is accessible via the database.
    """
    grid_search_result = list(agg.grid_searches())[0]["result"]
    print(
        f"****Best result (grid_search_result.best_samples)****\n\n {grid_search_result.best_samples}\n"
    )
    print(
        f"****Grid Log Evidences (grid_search_result.log_evidences().native)****\n\n {grid_search_result.log_evidences().native}\n"
    )

    """
    From the GridSearch, get an aggregator which contains only the maximum log likelihood model. E.g. if the 10th out of the 
    16 cells was the best fit:
    """
    print("\n\n****MAX LH AGGREGATOR VIA GRID****\n\n")

    print(
        f"Max LH Instance (agg_best_fit.values('instance')[0]) {agg.grid_searches().best_fits().values('instance')[0]}\n"
    )
    print(
        f"Max LH samples (agg_best_fit.values('samples')[0]) {agg.grid_searches().best_fits().values('samples')[0]}"
    )

    assert (
        agg.grid_searches().best_fits().values("samples")[0].log_likelihood_list[-1]
        > -1e88
    )

    """
    __Subhalo__
    """
    agg_subhalo = agg.query(agg.search.unique_tag == dataset_name)

    agg_no_subhalo = agg.query(agg_subhalo.search.name == "subhalo[1]")
    agg_subhalo_grid = agg.grid_searches()
    agg_subhalo_grid_best_fit = agg_subhalo_grid.best_fits()
    agg_with_subhalo = agg.query(
        agg_subhalo.search.name == "subhalo[3]_[single_plane_refine]"
    )

    print("\n\n***********************")
    print("**AGG SUBHALO TESTS**")
    print("***********************\n\n")

    fit_no_subhalo_agg = al.agg.FitImagingAgg(aggregator=agg_no_subhalo)
    fit_no_subhalo_gen = fit_no_subhalo_agg.max_log_likelihood_gen_from()
    fit_no_subhalo = list(fit_no_subhalo_gen)[0][0]

    fit_subhalo_grid_best_fit_agg = al.agg.FitImagingAgg(
        aggregator=agg_subhalo_grid_best_fit
    )
    fit_subhalo_grid_best_fit_gen = (
        fit_subhalo_grid_best_fit_agg.max_log_likelihood_gen_from()
    )
    fit_subhalo_grid_best_fit = list(fit_subhalo_grid_best_fit_gen)[0]

    fit_with_subhalo_agg = al.agg.FitImagingAgg(aggregator=agg_with_subhalo)
    fit_with_subhalo_gen = fit_with_subhalo_agg.max_log_likelihood_gen_from()
    fit_with_subhalo = list(fit_with_subhalo_gen)[0]

    info = list(agg_subhalo_grid_best_fit.values("info"))[0]

    grid_search_result = list(agg_subhalo_grid)[0]["result"]

    result = al.subhalo.SubhaloGridSearchResult(
        result=grid_search_result,
    )

    fit_imaging_with_subhalo = analysis.fit_from(
        instance=result.best_samples.max_log_likelihood(),
    )

    output = aplt.Output(
        #       path=result.search.paths.output_path,
        path=".",
        format="png",
    )

    mat_plot = aplt.MatPlot2D(
        output=output,
    )

    subhalo_plotter = al.subhalo.SubhaloPlotter(
        result=result,
        fit_imaging_no_subhalo=fit_no_subhalo,
        fit_imaging_with_subhalo=fit_imaging_with_subhalo,
        mat_plot_2d=mat_plot,
    )

    samples_no_subhalo = list(agg_no_subhalo.values("samples"))[0]

    subhalo_plotter.figure_figures_of_merit_grid(
        use_log_evidences=True,
        relative_to_value=samples_no_subhalo.log_evidence,
        remove_zeros=True,
    )

    subhalo_plotter.figure_mass_grid()
    subhalo_plotter.subplot_detection_imaging()
    subhalo_plotter.subplot_detection_fits()

    os.environ["PYAUTOFIT_TEST_MODE"] = "0"


if __name__ == "__main__":
    fit()
