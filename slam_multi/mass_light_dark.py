import autofit as af
import autolens as al

from . import slam_util

from typing import Optional, Union


def run(
    settings_search: af.SettingsSearch,
    analysis_list: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    lp_chain_tracer: al.Tracer,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    light_result: af.Result,
    dark: Optional[af.Model] = af.Model(al.mp.NFWMCRLudlow),
    smbh: Optional[af.Model] = None,
    use_gradient: bool = False,
    link_mass_to_light_ratios: bool = True,
    extra_galaxies: Optional[af.Collection] = None,
    dataset_model: Optional[af.Model] = None,
) -> af.Result:
    """
    The SLaM MASS LIGHT DARK PIPELINE, which fits a mass model where the stellar mass is modeled in a way linked
    to the stellar light alongside a dark matter halo.

    Parameters
    ----------
    settings_search
        The settings used to set up the non-linear search which are general to all SLaM pipelines, for example
        the `path_prefix`.
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    source_result_for_lens
        The result of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline,
        used for initializing model components associated with the lens galaxy.
    source_result_for_source
        The result of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline,
        used for initializing model components associated with the source galaxy.
    light_result
        The result of the SLaM LIGHT LP PIPELINE which ran before this pipeline.
    lens_bulge
        The model used to represent the light and mass distribution of the lens galaxy's bulge (set to
        None to omit a bulge).
    lens_disk
        The model used to represent the light and mass distribution of the lens galaxy's disk (set to
        None to omit a disk).
    smbh
        The `MassProfile` used to fit the a super massive black hole in the lens galaxy.
    lens_bulge
        The `LightMassProfile` `Model` used to represent the light and stellar mass distribution of the lens galaxy's
        bulge (set to None to omit a bulge).
    lens_disk
        The `LightMassProfile` `Model` used to represent the light and stellar mass distribution of the lens galaxy's
        disk (set to None to omit a disk).
    dark
        The `MassProfile` `Model` used to represent the dark matter distribution of the lens galaxy's (set to None to
        omit dark matter).
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__
    
    Search 1 of the MASS LIGHT DARK PIPELINE fits a lens model where:
    
     - The lens galaxy light and stellar mass is modeled using light and mass profiles [Priors on light model parameters
     initialized from LIGHT PIPELINE].
     - The lens galaxy dark mass is modeled using a dark mass distribution [No prior initialization].
     - The source galaxy's light is parametric or a pixelization depending on the previous pipeline [Model and priors 
     initialized from SOURCE PIPELINE].
     
    This search aims to accurately estimate the lens mass model, using the improved mass model priors and source model 
    of the SOURCE PIPELINE and LIGHT PIPELINE.
    
    The `mass_to_light_ratio` prior of each light and stellar profile is set using the Einstein Mass estimate of the
    SOURCE PIPELINE, specifically using values which are 1% and 500% this estimate.
    
    The dark matter mass profile has the lens and source redshifts added to it, which are used to determine its mass
    from the mass-to-concentration relation of Ludlow et al.    
    """
    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):

        lens_bulge = al.util.chaining.mass_light_dark_from(
            light_result=light_result[i],
            lp_chain_tracer=lp_chain_tracer,
            name="bulge",
            use_gradient=use_gradient,
        )
        lens_disk = al.util.chaining.mass_light_dark_from(
            light_result=light_result[i],
            lp_chain_tracer=lp_chain_tracer,
            name="disk",
            use_gradient=use_gradient,
        )

        lens_bulge, lens_disk = al.util.chaining.link_ratios(
            link_mass_to_light_ratios=link_mass_to_light_ratios,
            light_result=light_result[i],
            bulge=lens_bulge,
            disk=lens_disk,
        )

        if dark is not None:
            try:
                dark.centre = lens_bulge.centre
            except AttributeError:
                dark.centre = lens_bulge.profile_list[0].centre

            dark.mass_at_200 = af.LogUniformPrior(lower_limit=1e10, upper_limit=1e15)
            dark.redshift_object = light_result[i].instance.galaxies.lens.redshift
            dark.redshift_source = light_result[i].instance.galaxies.source.redshift

        if smbh is not None:
            smbh.centre = lens_bulge.centre

        source = al.util.chaining.source_from(
            result=source_result_for_source[i],
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=light_result[i].instance.galaxies.lens.redshift,
                    bulge=lens_bulge,
                    disk=lens_disk,
                    point=light_result[i].instance.galaxies.lens.point,
                    dark=dark,
                    shear=source_result_for_lens[i].model.galaxies.lens.shear,
                    smbh=smbh,
                ),
                source=source,
            ),
            extra_galaxies=extra_galaxies,
            dataset_model=dataset_model,
        )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list)

    search = af.DynestyStatic(
        name="mass_light_dark[1]",
        **settings_search.search_dict,
        nlive=250,
    )

    result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

    return result
