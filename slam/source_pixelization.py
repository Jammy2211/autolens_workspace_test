import autofit as af
import autolens as al
from . import slam_util
from . import extensions

from typing import Callable, Union, Optional


def no_lens_light(
    settings_autofit: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_hyper: al.SetupHyper,
    source_parametric_results: af.ResultsCollection,
    mesh: af.Model(al.AbstractMesh) = af.Model(al.mesh.DelaunayBrightnessImage),
    regularization: af.Model(al.AbstractRegularization) = af.Model(
        al.reg.AdaptiveBrightnessSplit
    ),
    multi_func: Optional[Callable] = None,
) -> af.ResultsCollection:
    """
    The SLaM SOURCE PIXELIZED PIPELINE for fitting imaging data without a lens light component.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    source_parametric_results
        The results of the SLaM SOURCE PARAMETRIC PIPELINE which ran before this pipeline.
    pixelization
        The pixelization used by the `Inversion` which fits the source light.
    regularization
        The regularization used by the `Inversion` which fits the source light.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the SOURCE PIXELIZED PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of SOURCE PARAMETRIC 
     PIPELINE].
     - The source galaxy's light is a `VoronoiMagnification` pixelization and `Constant` regularization scheme.

    This search aims to quickly estimate values for the pixelization resolution and regularization coefficient.
    """

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_parametric_results.last.instance.galaxies.lens.redshift,
                mass=source_parametric_results.last.instance.galaxies.lens.mass,
                shear=source_parametric_results.last.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_parametric_results.last.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=al.mesh.VoronoiMagnification,
                    regularization=al.reg.Constant,
                ),
                hyper_galaxy=setup_hyper.hyper_galaxy_source_from(
                    result=source_parametric_results.last
                ),
            ),
        ),
        clumps=slam_util.clumps_from(result=source_parametric_results.last),
        hyper_image_sky=setup_hyper.hyper_image_sky_from(
            result=source_parametric_results.last, as_model=False
        ),
        hyper_background_noise=setup_hyper.hyper_background_noise_from(
            result=source_parametric_results.last
        ),
    )

    if multi_func is not None:
        analysis = multi_func(analysis, model, index=0)

    search = af.DynestyStatic(
        name="source_pixelization[1]_mass[fixed]_source[pixelization_magnification_initialization]",
        **settings_autofit.search_dict,
        nlive=30,
    )

    result_1 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    In search 2 of the SOURCE PIXELIZED PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the 
     SOURCE PARAMETRIC PIPELINE].
     - The source galaxy's light is a `VoronoiMagnification` pixelization and `Constant` regularization scheme 
     [parameters are fixed to the result of search 1].

    This search aims to improve the lens mass model using the search 1 `Inversion`.
    """

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_1.instance.galaxies.lens.redshift,
                mass=source_parametric_results.last.model.galaxies.lens.mass,
                shear=source_parametric_results.last.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_1.instance.galaxies.source.redshift,
                pixelization=result_1.instance.galaxies.source.pixelization,
                hyper_galaxy=result_1.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        clumps=slam_util.clumps_from(
            result=source_parametric_results.last, mass_as_model=True
        ),
        hyper_image_sky=result_1.instance.hyper_image_sky,
        hyper_background_noise=result_1.instance.hyper_background_noise,
    )

    search = af.DynestyStatic(
        name="source_pixelization[2]_mass[total]_source[fixed]",
        **settings_autofit.search_dict,
        nlive=50,
    )

    result_2 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Model + Search + Analysis + Model-Fit (Search 3)__

    In search 3 of the SOURCE PIXELIZED PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of search 2].
     - The source galaxy's light is the input pixelization and regularization.

    This search aims to estimate values for the pixelization and regularization scheme.
    """

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_2.instance.galaxies.lens.redshift,
                mass=result_2.instance.galaxies.lens.mass,
                shear=result_2.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_2.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization, mesh=mesh, regularization=regularization
                ),
                hyper_galaxy=result_2.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        clumps=slam_util.clumps_from(result=result_2),
        hyper_image_sky=result_2.instance.hyper_image_sky,
        hyper_background_noise=result_2.instance.hyper_background_noise,
    )

    search = af.DynestyStatic(
        name="source_pixelization[3]_mass[fixed]_source[pixelization_initialization]",
        **settings_autofit.search_dict,
        nlive=30,
        dlogz=10.0,
        sample="rstagger",
    )

    analysis.set_hyper_dataset(result=result_2)

    if multi_func is not None:
        analysis = multi_func(analysis, model, index=1)

    result_3 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Model + Search + Analysis + Model-Fit (Search 4)__

    In search 4 of the SOURCE PIXELIZED PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the 
     search 2].
     - The source galaxy's light is the input pixelization and regularization scheme [parameters fixed to the result 
     of search 3].

    This search aims to improve the lens mass model using the input `Inversion`.
    """
    mass = slam_util.mass__from(
        mass=result_2.model.galaxies.lens.mass,
        result=source_parametric_results.last,
        unfix_mass_centre=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_3.instance.galaxies.lens.redshift,
                mass=mass,
                shear=result_2.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_3.instance.galaxies.source.redshift,
                pixelization=result_3.instance.galaxies.source.pixelization,
                hyper_galaxy=result_3.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        clumps=slam_util.clumps_from(result=result_2, mass_as_model=True),
        hyper_image_sky=result_3.instance.hyper_image_sky,
        hyper_background_noise=result_3.instance.hyper_background_noise,
    )

    search = af.DynestyStatic(
        name="source_pixelization[4]_mass[total]_source[fixed]",
        **settings_autofit.search_dict,
        nlive=50,
    )

    result_4 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Hyper Extension__

    The above search is extended with a hyper-search if the SetupHyper has one or more of the following inputs:

     - The source is modeled using a pixelization with a regularization scheme.
     - One or more `HyperGalaxy`'s are included.
     - The background sky is included via `hyper_image_sky` input.
     - The background noise is included via the `hyper_background_noise`.
    """
    result_4 = extensions.hyper_fit(
        setup_hyper=setup_hyper,
        result=result_4,
        analysis=analysis,
        search_previous=search,
        include_hyper_image_sky=True,
    )

    return af.ResultsCollection([result_1, result_2, result_3, result_4])


def with_lens_light(
    settings_autofit: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_hyper: al.SetupHyper,
    source_parametric_results: af.ResultsCollection,
    mesh: af.Model(al.AbstractMesh) = af.Model(al.mesh.DelaunayBrightnessImage),
    regularization: af.Model(al.AbstractRegularization) = af.Model(
        al.reg.AdaptiveBrightnessSplit
    ),
    multi_func: Optional[Callable] = None,
) -> af.ResultsCollection:
    """
    The SLaM SOURCE PIXELIZED PIPELINE for fitting imaging data with a lens light component.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    source_parametric_results
        The results of the SLaM SOURCE PARAMETRIC PIPELINE which ran before this pipeline.
    pixelization
        The pixelization used by the `Inversion` which fits the source light.
    regularization
        The regularization used by the `Inversion` which fits the source light.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the SOURCE PIXELIZED PIPELINE we fit a lens model where:

    - The lens galaxy light is modeled using a parametric bulge + disk + envelope [parameters fixed to result of SOURCE
    PARAMETER PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of SOURCE PARAMETRIC 
     PIPELINE].
     - The source galaxy's light is a `VoronoiMagnification` pixelization and `Constant` regularization scheme.

    This search aims to quickly estimate values for the pixelization resolution and regularization coefficient.
    """

    instance = source_parametric_results.last.instance
    fit = source_parametric_results.last.max_log_likelihood_fit

    bulge = slam_util.lp_from(lp=instance.galaxies.lens.bulge, fit=fit)
    disk = slam_util.lp_from(lp=instance.galaxies.lens.disk, fit=fit)
    envelope = slam_util.lp_from(lp=instance.galaxies.lens.envelope, fit=fit)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_parametric_results.last.instance.galaxies.lens.redshift,
                bulge=bulge,
                disk=disk,
                envelope=envelope,
                mass=source_parametric_results.last.instance.galaxies.lens.mass,
                shear=source_parametric_results.last.instance.galaxies.lens.shear,
                hyper_galaxy=setup_hyper.hyper_galaxy_lens_from(
                    result=source_parametric_results.last
                ),
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_parametric_results.last.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=al.mesh.VoronoiMagnification,
                    regularization=al.reg.Constant,
                ),
                hyper_galaxy=setup_hyper.hyper_galaxy_source_from(
                    result=source_parametric_results.last
                ),
            ),
        ),
        clumps=slam_util.clumps_from(result=source_parametric_results.last),
        hyper_image_sky=setup_hyper.hyper_image_sky_from(
            result=source_parametric_results.last, as_model=False
        ),
        hyper_background_noise=setup_hyper.hyper_background_noise_from(
            result=source_parametric_results.last
        ),
    )

    if multi_func is not None:
        analysis = multi_func(analysis, model, index=0)

    search = af.DynestyStatic(
        name="source_pixelization[1]_light[fixed]_mass[fixed]_source[pixelization_magnification_initialization]",
        **settings_autofit.search_dict,
        nlive=30,
    )

    result_1 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    In search 2 of the SOURCE PIXELIZED PIPELINE we fit a lens model where:

    - The lens galaxy light is modeled using a parametric bulge + disk + envelope [parameters fixed to result of SOURCE
    PARAMETER PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the 
     SOURCE PARAMETRIC PIPELINE].
     - The source galaxy's light is a `VoronoiMagnification` pixelization and `Constant` regularization scheme 
     [parameters are fixed to the result of search 1].

    This search aims to improve the lens mass model using the search 1 `Inversion`.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_1.instance.galaxies.lens.redshift,
                bulge=result_1.instance.galaxies.lens.bulge,
                disk=result_1.instance.galaxies.lens.disk,
                envelope=result_1.instance.galaxies.lens.envelope,
                mass=source_parametric_results.last.model.galaxies.lens.mass,
                shear=source_parametric_results.last.model.galaxies.lens.shear,
                hyper_galaxy=result_1.instance.galaxies.lens.hyper_galaxy,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_1.instance.galaxies.source.redshift,
                pixelization=result_1.instance.galaxies.source.pixelization,
                hyper_galaxy=result_1.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        clumps=slam_util.clumps_from(
            result=source_parametric_results.last, mass_as_model=True
        ),
        hyper_image_sky=result_1.instance.hyper_image_sky,
        hyper_background_noise=result_1.instance.hyper_background_noise,
    )

    search = af.DynestyStatic(
        name="source_pixelization[2]_light[fixed]_mass[total]_source[pixelization_magnification]",
        **settings_autofit.search_dict,
        nlive=50,
    )

    result_2 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Model + Search + Analysis + Model-Fit (Search 3)__

    In search 3 of the SOURCE PIXELIZED PIPELINE we fit a lens model where:

    - The lens galaxy light is modeled using a parametric bulge + disk + envelope [parameters fixed to result of SOURCE
    PARAMETER PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of search 2].
     - The source galaxy's light is the input pixelization and regularization.

    This search aims to estimate values for the pixelization and regularization scheme.
    """

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_2.instance.galaxies.lens.redshift,
                bulge=result_2.instance.galaxies.lens.bulge,
                disk=result_2.instance.galaxies.lens.disk,
                envelope=result_2.instance.galaxies.lens.envelope,
                mass=result_2.instance.galaxies.lens.mass,
                shear=result_2.instance.galaxies.lens.shear,
                hyper_galaxy=result_2.instance.galaxies.lens.hyper_galaxy,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_2.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization, mesh=mesh, regularization=regularization
                ),
                hyper_galaxy=result_2.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        clumps=slam_util.clumps_from(result=result_2),
        hyper_image_sky=result_2.instance.hyper_image_sky,
        hyper_background_noise=result_2.instance.hyper_background_noise,
    )

    if multi_func is not None:
        analysis = multi_func(analysis, model, index=1)

    search = af.DynestyStatic(
        name="source_pixelization[3]_light[fixed]_mass[fixed]_source[pixelization_initialization]",
        **settings_autofit.search_dict,
        nlive=30,
        dlogz=10.0,
        sample="rstagger",
    )

    analysis.set_hyper_dataset(result=result_2)

    result_3 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Model + Search + Analysis + Model-Fit (Search 4)__

    In search 4 of the SOURCE PIXELIZED PIPELINE we fit a lens model where:

    - The lens galaxy light is modeled using a parametric bulge + disk + envelope [parameters fixed to result of SOURCE
    PARAMETER PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the 
     search 2].
     - The source galaxy's light is the input pixelization and regularization scheme [parameters fixed to the result 
     of search 3].

    This search aims to improve the lens mass model using the input `Inversion`.
    """
    mass = slam_util.mass__from(
        mass=result_2.model.galaxies.lens.mass,
        result=source_parametric_results.last,
        unfix_mass_centre=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_3.instance.galaxies.lens.redshift,
                bulge=result_3.instance.galaxies.lens.bulge,
                disk=result_3.instance.galaxies.lens.disk,
                envelope=result_3.instance.galaxies.lens.envelope,
                mass=mass,
                shear=result_2.model.galaxies.lens.shear,
                hyper_galaxy=result_3.instance.galaxies.lens.hyper_galaxy,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_3.instance.galaxies.source.redshift,
                pixelization=result_3.instance.galaxies.source.pixelization,
                hyper_galaxy=result_3.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        clumps=slam_util.clumps_from(result=result_2, mass_as_model=True),
        hyper_image_sky=result_3.instance.hyper_image_sky,
        hyper_background_noise=result_3.instance.hyper_background_noise,
    )

    search = af.DynestyStatic(
        name="source_pixelization[4]_light[fixed]_mass[total]_source[pixelization]",
        **settings_autofit.search_dict,
        nlive=50,
    )

    result_4 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Hyper Extension__

    The above search is extended with a hyper-search if the SetupHyper has one or more of the following inputs:

     - The source is modeled using a pixelization with a regularization scheme.
     - One or more `HyperGalaxy`'s are included.
     - The background sky is included via `hyper_image_sky` input.
     - The background noise is included via the `hyper_background_noise`.
    """
    result_4 = extensions.hyper_fit(
        setup_hyper=setup_hyper,
        result=result_4,
        analysis=analysis,
        search_previous=search,
        include_hyper_image_sky=True,
    )

    return af.ResultsCollection([result_1, result_2, result_3, result_4])