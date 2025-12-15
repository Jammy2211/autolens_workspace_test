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

    import numpy as np
    import os
    from os import path

    import autofit as af
    import autolens as al
    import autolens.plot as aplt
    import slam_pipeline

    """
    __Search Settings__
    """
    settings_search = af.SettingsSearch(
        path_prefix=path.join("model_graph"),
        number_of_cores=1,
        session=None,
    )

    """
    __Dataset__ 
    """
    dataset_waveband_list = ["g", "r"]
    pixel_scale_list = [0.12, 0.08]

    dataset_name = "lens_sersic"
    dataset_main_path = path.join("dataset", "multi", dataset_name)

    dataset_list = []

    for dataset_waveband, pixel_scale in zip(dataset_waveband_list, pixel_scale_list):
        dataset = al.Imaging.from_fits(
            data_path=path.join(dataset_main_path, f"{dataset_waveband}_data.fits"),
            noise_map_path=path.join(
                dataset_main_path, f"{dataset_waveband}_noise_map.fits"
            ),
            psf_path=path.join(dataset_main_path, f"{dataset_waveband}_psf.fits"),
            pixel_scales=pixel_scale,
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
            radial_list=[0.3, 0.6],
            centre_list=[(0.0, 0.0)],
        )

        dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        dataset_list.append(dataset)

    """
    __Model 1__
    """
    redshift_lens = 0.5
    redshift_source = 1.0

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

    """
    __Analysis List__
    """
    analysis_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

    """
    __Preamble__
    """
    lens_bulge = lens_bulge
    lens_disk = af.Model(al.lp.Exponential)
    lens_point = None
    mass: af.Model = af.Model(al.mp.Isothermal)
    shear: af.Model(al.mp.ExternalShear) = af.Model(al.mp.ExternalShear)
    source_bulge = source_bulge
    source_disk = None
    extra_galaxies = None
    dataset_model = af.Model(al.DatasetModel)

    """
    __Analysis Graphical Model__
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=lens_disk,
                point=lens_point,
                mass=mass,
                shear=shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=source_bulge,
                disk=source_disk,
            ),
        ),
        extra_galaxies=extra_galaxies,
        dataset_model=dataset_model,
    )

    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):

        model_analysis = model.copy()

        if i > 0:
            model_analysis.dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
                lower_limit=-1.0, upper_limit=1.0
            )
            model_analysis.dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
                lower_limit=-1.0, upper_limit=1.0
            )

        analysis_factor = af.AnalysisFactor(
            prior_model=model_analysis, analysis=analysis
        )

        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list)

    search = af.DynestyStatic(
        name="task_5_analysis_graph_database",
        **settings_search.search_dict,
        nlive=200,
        maxcall=500,
        maxiter=500,
    )

    source_lp_result = search.fit(
        model=factor_graph.global_prior_model,
        analysis=factor_graph,
        info={"hi": "there"},
    )

    """
    __Database Stuff__
    """

    from autofit.database.aggregator import Aggregator

    database_file = "task_5_database_graph.sqlite"

    try:
        os.remove(path.join("output", database_file))
    except FileNotFoundError:
        pass

    agg = Aggregator.from_database(database_file)
    agg.add_directory(
        directory=path.join("output", "model_graph", "task_5_analysis_graph_database")
    )

    assert len(agg) > 0

    """
    __Samples + Results__

    Make sure database + agg can be used.
    """
    for samples in agg.values("samples"):
        print(samples.log_likelihood_list)
        print(samples.log_likelihood_list[0])

    ml_vector = [
        samps.max_log_likelihood(as_instance=False) for samps in agg.values("samples")
    ]
    print(ml_vector, "\n\n")

    """
    __Queries__
    """
    unique_tag = agg.search.unique_tag
    agg_query = agg.query(unique_tag == "mass_sie__source_sersic__1")
    samples_gen = agg_query.values("samples")

    unique_tag = agg.search.unique_tag
    agg_query = agg.query(unique_tag == "incorrect_name")
    samples_gen = agg_query.values("samples")

    name = agg.search.name
    agg_query = agg.query(name == "database_example")
    print("Total Queried Results via search name = ", len(agg_query), "\n\n")

    lens = agg.model.galaxies.lens
    agg_query = agg.query(lens.mass == al.mp.Isothermal)
    print("Total Samples Objects via `Isothermal` model query = ", len(agg_query), "\n")

    source = agg.model.galaxies.source
    agg_query = agg.query(source.disk == None)
    print("Total Samples Objects via `Isothermal` model query = ", len(agg_query), "\n")

    mass = agg.model.galaxies.lens.mass
    agg_query = agg.query((mass == al.mp.Isothermal) & (mass.einstein_radius > 1.0))
    print(
        "Total Samples Objects In Query `Isothermal and einstein_radius > 3.0` = ",
        len(agg_query),
        "\n",
    )

    """
    __Files__

    Check that all other files stored in database (e.g. model, search) can be loaded and used.
    """

    for model in agg.values("model"):
        print(model.info)

    for search in agg.values("search"):
        print(search)

    for samples_summary in agg.values("samples_summary"):
        instance = samples_summary.max_log_likelihood()
        print(instance[0].galaxies)

    for info in agg.values("info"):
        print(info["hi"])

    for dataset in agg.child_values("dataset"):
        print(dataset)

    try:
        for covariance in agg.values("covariance"):
            print(covariance)
    except ValueError:
        pass

    """
    __Aggregator Module__
    """
    tracer_agg = al.agg.TracerAgg(aggregator=agg)
    tracer_gen = tracer_agg.max_log_likelihood_gen_from()

    grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

    for tracer_list in tracer_gen:
        # Only one `Analysis` so take first and only tracer.
        tracer = tracer_list[0]

        tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
        tracer_plotter.figures_2d(convergence=True, potential=True)

        print("Tracer Checked")

    imaging_agg = al.agg.ImagingAgg(aggregator=agg)
    imaging_gen = imaging_agg.dataset_gen_from()

    for dataset_list in imaging_gen:
        dataset = dataset_list[0]

        dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
        dataset_plotter.subplot_dataset()

        print("Imaging Checked")

    fit_agg = al.agg.FitImagingAgg(
        aggregator=agg,
        settings_inversion=al.SettingsInversion(use_border_relocator=False),
    )
    fit_imaging_gen = fit_agg.max_log_likelihood_gen_from()

    for fit_list in fit_imaging_gen:
        fit = fit_list[0]

        print(fit.dataset_model.grid_offset.grid_offset_1)

        print(fit_list[0].dataset_model.grid_offset.grid_offset_1)
        print(fit_list[1].dataset_model.grid_offset)

        fit_plotter = aplt.FitImagingPlotter(fit=fit)
        fit_plotter.subplot_fit()

        print("FitImaging Checked")

    fit_agg = al.agg.FitImagingAgg(
        aggregator=agg,
        settings_inversion=al.SettingsInversion(use_border_relocator=False),
    )
    fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=3)

    i = 0

    for fit_gen in fit_pdf_gen:  # 1 Dataset so just one fit

        for (
            fit_list
        ) in (
            fit_gen
        ):  # Iterate over each total_samples=3, each with two fits for each analysis.

            i += 1

            print(fit_list[0].dataset_model.grid_offset.grid_offset_1)
            print(fit_list[1].dataset_model.grid_offset)

            fit_plotter = aplt.FitImagingPlotter(fit=fit_list[0])
            fit_plotter.subplot_fit()

            fit_plotter = aplt.FitImagingPlotter(fit=fit_list[1])
            fit_plotter.subplot_fit()

    assert i == 3


if __name__ == "__main__":
    fit()
