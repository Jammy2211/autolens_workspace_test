import autofit as af
import autolens as al
from . import slam_util
from . import extensions

from typing import Union, Optional, Tuple


def run(
    settings_autofit: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_adapt: al.SetupAdapt,
    source_results: af.ResultsCollection,
    light_results: Optional[af.ResultsCollection],
    mass: af.Model = af.Model(al.mp.Isothermal),
    smbh: Optional[af.Model] = None,
    mass_centre: Optional[Tuple[float, float]] = None,
    end_with_adapt_extension: bool = False,
) -> af.ResultsCollection:
    """
    The SLaM MASS TOTAL PIPELINE for fitting imaging data with a lens light component.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_adapt
        The setup of the adapt fit.
    source_results
        The results of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline.
    light_results
        The results of the SLaM LIGHT LP PIPELINE which ran before this pipeline.
    mass
        The `MassProfile` used to fit the lens galaxy mass in this pipeline.
    smbh
        The `MassProfile` used to fit the a super massive black hole in the lens galaxy.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    end_with_adapt_extension
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
        mass=mass, result=source_results[0], unfix_mass_centre=True
    )

    if mass_centre is not None:
        mass.centre = mass_centre

    if smbh is not None:
        smbh.centre = mass.centre

    if light_results is None:

        bulge = None
        disk = None
        point = None

    else:

        bulge = light_results.last.instance.galaxies.lens.bulge
        disk = light_results.last.instance.galaxies.lens.disk
        point = light_results.last.instance.galaxies.lens.point

    source = slam_util.source__from_result_model_if_parametric(
        result=source_results.last,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_results.last.instance.galaxies.lens.redshift,
                bulge=bulge,
                disk=disk,
                point=point,
                mass=mass,
                shear=source_results[0].model.galaxies.lens.shear,
                smbh=smbh,
            ),
            source=source,
        ),
        clumps=slam_util.clumps_from(result=source_results[0], mass_as_model=True),
    )

    search = af.DynestyStatic(
        name="mass_total[1]_light[lp]_mass[total]_source",
        **settings_autofit.search_dict,
        nlive=150,
    )

    result_1 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Hyper Extension__

    The above search may be extended with a hyper-search, if the SetupAdapt has one or more of the following inputs:

     - The source is modeled using a pixelization with a regularization scheme.
    """

    if end_with_adapt_extension:

        result_1 = extensions.adapt_fit(
            setup_adapt=setup_adapt,
            result=result_1,
            analysis=analysis,
            search_previous=search,
        )

    return af.ResultsCollection([result_1])
