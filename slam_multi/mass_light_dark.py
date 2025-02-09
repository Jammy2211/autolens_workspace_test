import autofit as af
import autolens as al


from typing import Optional, Union


def run(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    light_result: af.Result,
    lens_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    lens_disk: Optional[af.Model] = None,
    dark: af.Model = af.Model(al.mp.NFWMCRLudlow),
    smbh: Optional[af.Model] = None,
) -> af.ResultsCollection:
    """
    The SLaM MASS LIGHT DARK PIPELINE, which fits a mass model where the stellar mass is modeled in a way linked
    to the stellar light alongside a dark matter halo.

    Parameters
    ----------
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

    lens_bulge = al.util.chaining.mass_light_dark_from(
        lmp_model=lens_bulge,
        result_light_component=light_result.model.galaxies.lens.bulge,
    )
    lens_disk = al.util.chaining.mass_light_dark_from(
        lmp_model=lens_disk,
        result_light_component=light_result.model.galaxies.lens.disk,
    )
    # lens_point = al.util.chaining.mass_light_dark_from(
    #     lmp_model=lens_point,
    #     result_light_component=light_result.model.galaxies.lens.point,
    # )

    dark.mass_at_200 = af.LogUniformPrior(lower_limit=1e10, upper_limit=1e15)
    dark.redshift_object = light_result.instance.galaxies.lens.redshift
    dark.redshift_source = light_result.instance.galaxies.source.redshift

    if smbh is not None:
        smbh.centre = lens_bulge.centre

    source = al.util.chaining.source_from(
        result=source_result_for_source,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=light_result.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=lens_disk,
                #    point=lens_point,
                dark=dark,
                shear=source_result_for_lens.model.galaxies.lens.shear,
                smbh=smbh,
            ),
            source=source,
        ),
        extra_galaxies=al.util.chaining.extra_galaxies_from(
            result=source_result_for_lens, mass_as_model=True
        ),
    )

    search = af.Nautilus(
        name="mass_light_dark[1]_light[lp]_mass[light_dark]_source",
        **settings_search.search_dict,
        n_live=150,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result
