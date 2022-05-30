import autofit as af
import autolens as al
from . import slam_util
from . import extensions

from typing import Union, Optional, Tuple


def no_lens_light(
    settings_autofit: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_hyper: al.SetupHyper,
    source_results: af.ResultsCollection,
    mass: af.Model = af.Model(al.mp.EllIsothermal),
    smbh: Optional[af.Model] = None,
    mass_centre: Optional[Tuple[float, float]] = None,
    end_with_hyper_extension: bool = False,
    end_with_stochastic_extension: bool = False
) -> af.ResultsCollection:
    """
    The SLaM MASS TOTAL PIPELINE for fitting imaging data without a lens light component.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    source_results
        The results of the SLaM SOURCE PARAMETRIC PIPELINE or SOURCE PIXELIZED PIPELINE which ran before this pipeline.
    mass
        The `MassProfile` used to fit the lens galaxy mass in this pipeline.
    smbh
        The `MassProfile` used to fit the a super massive black hole in the lens galaxy.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    end_with_hyper_extension
        If `True` a hyper extension is performed at the end of the pipeline. If this feature is used, you must be
        certain you have manually passed the new hyper images geneted in this search to the next pipelines.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the MASS TOTAL PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [Priors initialized from SOURCE PIPELINE].
     - The source galaxy's light is parametric or an inversion depending on the previous pipeline [Model and priors 
     initialized from SOURCE PIPELINE].

    This search aims to accurately estimate the lens mass model, using the improved mass model priors and source model 
    of the SOURCE PIPELINE
    """
    mass = slam_util.mass__from(
        mass=mass, result=source_results.last, unfix_mass_centre=True
    )

    if mass_centre is not None:
        mass.centre = mass_centre

    if smbh is not None:
        smbh.centre = mass.centre

    source = slam_util.source__from_result_model_if_parametric(
        result=source_results.last, setup_hyper=setup_hyper
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_results.last.instance.galaxies.lens.redshift,
                mass=mass,
                smbh=smbh,
                shear=source_results.last.model.galaxies.lens.shear,
            ),
            source=source,
        ),
        clumps=slam_util.clumps_from(result=source_results.last, mass_as_model=True),
        hyper_image_sky=setup_hyper.hyper_image_sky_from(
            result=source_results.last, as_model=True
        ),
        hyper_background_noise=setup_hyper.hyper_background_noise_from(
            result=source_results.last
        ),
    )

    search = af.DynestyStatic(
        name="mass_total[1]_mass[total]_source",
        **settings_autofit.search_dict,
        nlive=100,
    )

    result_1 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Hyper Extension__

    The above search may be extended with a hyper-search, if the SetupHyper has one or more of the following inputs:

     - The source is modeled using a pixelization with a regularization scheme.
     - One or more `HyperGalaxy`'s are included.
     - The background sky is included via `hyper_image_sky` input.
     - The background noise is included via the `hyper_background_noise`.
    """

    if end_with_hyper_extension:

        result_1 = extensions.hyper_fit(
        setup_hyper=setup_hyper,
        result=result_1,
        analysis=analysis,
        search_previous=search,
        include_hyper_image_sky=True,
    )

    if end_with_stochastic_extension:

        extensions.stochastic_fit(
            result=result_1, analysis=analysis, search_previous=search, **settings_autofit.fit_dict
        )

    return af.ResultsCollection([result_1])


def with_lens_light(
    settings_autofit: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_hyper: al.SetupHyper,
    source_results: af.ResultsCollection,
    light_results: af.ResultsCollection,
    mass: af.Model = af.Model(al.mp.EllIsothermal),
    smbh: Optional[af.Model] = None,
    mass_centre: Optional[Tuple[float, float]] = None,
    end_with_hyper_extension: bool = False,
    end_with_stochastic_extension: bool = False
) -> af.ResultsCollection:
    """
    The SLaM MASS TOTAL PIPELINE for fitting imaging data with a lens light component.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    source_results
        The results of the SLaM SOURCE PARAMETRIC PIPELINE or SOURCE PIXELIZED PIPELINE which ran before this pipeline.
    light_results
        The results of the SLaM LIGHT PARAMETRIC PIPELINE which ran before this pipeline.
    mass
        The `MassProfile` used to fit the lens galaxy mass in this pipeline.
    smbh
        The `MassProfile` used to fit the a super massive black hole in the lens galaxy.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    end_with_hyper_extension
        If `True` a hyper extension is performed at the end of the pipeline. If this feature is used, you must be
        certain you have manually passed the new hyper images geneted in this search to the next pipelines.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the MASS TOTAL PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [Priors initialized from SOURCE PIPELINE].
     - The source galaxy's light is parametric or an inversion depending on the previous pipeline [Model and priors 
     initialized from SOURCE PIPELINE].

    This search aims to accurately estimate the lens mass model, using the improved mass model priors and source model 
    of the SOURCE PIPELINE
    """
    mass = slam_util.mass__from(
        mass=mass, result=source_results.last, unfix_mass_centre=True
    )

    if mass_centre is not None:
        mass.centre = mass_centre

    if smbh is not None:
        smbh.centre = mass.centre

    instance = light_results.last.instance
    fit = light_results.last.max_log_likelihood_fit

    bulge = slam_util.lp_from(lp=instance.galaxies.lens.bulge, fit=fit)
    disk = slam_util.lp_from(lp=instance.galaxies.lens.disk, fit=fit)
    envelope = slam_util.lp_from(lp=instance.galaxies.lens.envelope, fit=fit)

    lens_gaussian_dict = slam_util.gaussian_dict_lp_from(
        galaxy=instance.galaxies.lens, fit=fit
    )

    source = slam_util.source__from_result_model_if_parametric(
        result=light_results.last, setup_hyper=setup_hyper
    )

    if lens_gaussian_dict is None:

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=light_results.last.instance.galaxies.lens.redshift,
                    bulge=bulge,
                    disk=disk,
                    envelope=envelope,
                    mass=mass,
                    shear=source_results.last.model.galaxies.lens.shear,
                    smbh=smbh,
                    hyper_galaxy=setup_hyper.hyper_galaxy_lens_from(
                        result=light_results.last
                    ),
                ),
                source=source,
            ),
            clumps=slam_util.clumps_from(
                result=source_results.last, mass_as_model=True
            ),
            hyper_image_sky=setup_hyper.hyper_image_sky_from(
                result=light_results.last, as_model=True
            ),
            hyper_background_noise=setup_hyper.hyper_background_noise_from(
                result=light_results.last
            ),
        )

    else:

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=light_results.last.instance.galaxies.lens.redshift,
                    bulge=bulge,
                    disk=disk,
                    envelope=envelope,
                    **lens_gaussian_dict,
                    mass=mass,
                    shear=source_results.last.model.galaxies.lens.shear,
                    smbh=smbh,
                    hyper_galaxy=setup_hyper.hyper_galaxy_lens_from(
                        result=light_results.last
                    ),
                ),
                source=source,
            ),
            clumps=slam_util.clumps_from(
                result=source_results.last, mass_as_model=True
            ),
            hyper_image_sky=setup_hyper.hyper_image_sky_from(
                result=light_results.last, as_model=True
            ),
            hyper_background_noise=setup_hyper.hyper_background_noise_from(
                result=light_results.last
            ),
        )

    search = af.DynestyStatic(
        name="mass_total[1]_light[parametric]_mass[total]_source",
        **settings_autofit.search_dict,
        nlive=100,
    )

    result_1 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Hyper Extension__

    The above search may be extended with a hyper-search, if the SetupHyper has one or more of the following inputs:

     - The source is modeled using a pixelization with a regularization scheme.
     - One or more `HyperGalaxy`'s are included.
     - The background sky is included via `hyper_image_sky` input.
     - The background noise is included via the `hyper_background_noise`.
    """

    if end_with_hyper_extension:

        result_1 = extensions.hyper_fit(
        setup_hyper=setup_hyper,
        result=result_1,
        analysis=analysis,
        search_previous=search,
        include_hyper_image_sky=True,
    )

    if end_with_stochastic_extension:

        extensions.stochastic_fit(
            result=result_1, analysis=analysis, search_previous=search, **settings_autofit.fit_dict
        )

    return af.ResultsCollection([result_1])
