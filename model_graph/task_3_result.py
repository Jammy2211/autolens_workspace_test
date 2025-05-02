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
    import slam

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
            sub_size_list=[8, 4, 1],
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
    __Analysis Summing__

    With Analysis Summing, I can set up a model via prior passing as follows
    """
    analysis_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]
    analysis = sum(analysis_list)

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
        #    dataset_model=dataset_model,
    )

    #    analysis = analysis.with_free_parameters(model.dataset_model.grid_offset)

    search = af.DynestyStatic(
        name="task_2_analysis_summing_search_1",
        **settings_search.search_dict,
        nlive=200,
    )

    source_lp_result = search.fit(
        model=model, analysis=analysis, **settings_search.fit_dict
    )

    positions_likelihood = source_lp_result[0].positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    )

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
    analysis_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

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
        #    dataset_model=dataset_model,
    )

    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):

        model_analysis = model.copy()

        # if i > 0:
        #     model_analysis.dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
        #         lower_limit=-1.0, upper_limit=1.0
        #     )
        #     model_analysis.dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
        #         lower_limit=-1.0, upper_limit=1.0
        #     )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list)

    search = af.DynestyStatic(
        name="task_2_analysis_graph_search_1",
        **settings_search.search_dict,
        nlive=200,
    )

    source_lp_result = search.fit(
        model=factor_graph.global_prior_model, analysis=factor_graph
    )

    positions_likelihood = source_lp_result.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    )

    print(positions_likelihood)


if __name__ == "__main__":
    fit()
