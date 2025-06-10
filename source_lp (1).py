import autofit as af
import autolens as al
from autogalaxy.analysis import chaining_util

from typing import Union, Optional, Tuple


def run_1(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    lens_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    lens_disk: Optional[af.Model] = af.Model(al.lp.Exponential),
    lens_point: Optional[af.Model] = None,
    mass: af.Model = af.Model(al.mp.Isothermal),
    shear: af.Model(al.mp.ExternalShear) = af.Model(al.mp.ExternalShear),
    source_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    source_disk: Optional[af.Model] = None,
    redshift_lens: float = 0.5,
    redshift_source: float = 1.0,
    redshift_source_2: Optional[float] = None,
    mass_centre: Optional[Tuple[float, float]] = None,
    extra_galaxies: Optional[af.Collection] = None,
) -> af.Result:
    """
    The SlaM SOURCE LP PIPELINE, which provides an initial model for the lens's light, mass and source using a
    parametric source model (e.g. MGE, Sersics).

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    lens_bulge
        The model used to represent the light distribution of the lens galaxy's bulge (set to
        None to omit a bulge).
    lens_disk
        The model used to represent the light distribution of the lens galaxy's disk (set to
        None to omit a disk).
    lens_point
        The model used to represent the light distribution of point source emission in the lens galaxy (set to
        None to omit a point).
    mass
        The `MassProfile` fitted by this pipeline.
    shear
        The model used to represent the external shear in the mass model (set to None to turn off shear).
    source_bulge
        The model used to represent the light distribution of the source galaxy's bulge (set to
        None to omit a bulge).
    source_disk
        The model used to represent the light distribution of the source galaxy's disk (set to
        None to omit a disk).
    redshift_lens
        The redshift of the lens galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    redshift_source
        The redshift of the source galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the SOURCE LP PIPELINE fits a lens model where:

     - The lens galaxy light is modeled using a light profiles [no prior initialization].
     - The lens galaxy mass is modeled using a total mass distribution [no prior initialization].
     - The source galaxy's light is a light profiles [no prior initialization].

    This search aims to accurately estimate an initial lens light model, mass model and source model.
    """

    if mass_centre is not None:
        mass.centre = mass_centre

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
            source_2=af.Model(
                al.Galaxy,
                redshift=redshift_source_2,
            )
        ),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=200,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result

def run_2(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_lp_result: af.Result,
    extra_galaxies: Optional[af.Collection] = None,
) -> af.Result:
    """
    The SlaM SOURCE LP PIPELINE, which provides an initial model for the lens's light, mass and source using a
    parametric source model (e.g. MGE, Sersics).

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    lens_bulge
        The model used to represent the light distribution of the lens galaxy's bulge (set to
        None to omit a bulge).
    lens_disk
        The model used to represent the light distribution of the lens galaxy's disk (set to
        None to omit a disk).
    lens_point
        The model used to represent the light distribution of point source emission in the lens galaxy (set to
        None to omit a point).
    mass
        The `MassProfile` fitted by this pipeline.
    shear
        The model used to represent the external shear in the mass model (set to None to turn off shear).
    source_bulge
        The model used to represent the light distribution of the source galaxy's bulge (set to
        None to omit a bulge).
    source_disk
        The model used to represent the light distribution of the source galaxy's disk (set to
        None to omit a disk).
    redshift_lens
        The redshift of the lens galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    redshift_source
        The redshift of the source galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 2 of the SOURCE LP PIPELINE fits a lens model where:

     - The lens extra galaxy light is modeled using light profiles [no prior initialization].
     - The lens extra galaxy mass is modeled using a total mass distribution [no prior initialization].
     - The source galaxy's light is a light profiles [no prior initialization].
     - The main lens galaxy is fixed from SOURCE LP 1 [instance]

    This search aims to accurately estimate an initial lens light model, mass model and source model.
    """

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                point=source_lp_result.instance.galaxies.lens.point,
                mass=source_lp_result.instance.galaxies.lens.mass,
                shear=source_lp_result.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                bulge=source_lp_result.instance.galaxies.source.bulge,
                disk=source_lp_result.instance.galaxies.source.disk,
            ),
        ),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="source_lp[2]",
        **settings_search.search_dict,
        n_live=150,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result

def run_DSPL(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    lens_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    lens_mass: af.Model = af.Model(al.mp.Isothermal),
    shear: af.Model(al.mp.ExternalShear) = af.Model(al.mp.ExternalShear),
    source_bulge_1: Optional[af.Model] = af.Model(al.lp.Sersic),
    source_mass_1: Optional[af.Model] = af.Model(al.mp.Isothermal),
    source_bulge_2: Optional[af.Model] = af.Model(al.lp.Sersic),
    redshift_lens: float = 0.5,
    redshift_source_1: float = 1.0,
    redshift_source_2: float = 2.0,
    extra_galaxies: Optional[af.Collection] = None,
) -> af.Result:
    """
    The SlaM SOURCE LP PIPELINE, which provides an initial model for the lens's light, mass and source using a
    parametric source model (e.g. MGE, Sersics).

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    lens_bulge
        The model used to represent the light distribution of the lens galaxy's bulge (set to
        None to omit a bulge).
    lens_disk
        The model used to represent the light distribution of the lens galaxy's disk (set to
        None to omit a disk).
    lens_point
        The model used to represent the light distribution of point source emission in the lens galaxy (set to
        None to omit a point).
    mass
        The `MassProfile` fitted by this pipeline.
    shear
        The model used to represent the external shear in the mass model (set to None to turn off shear).
    source_bulge
        The model used to represent the light distribution of the source galaxy's bulge (set to
        None to omit a bulge).
    source_disk
        The model used to represent the light distribution of the source galaxy's disk (set to
        None to omit a disk).
    redshift_lens
        The redshift of the lens galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    redshift_source
        The redshift of the source galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 2 of the SOURCE LP PIPELINE fits a lens model where:

     - The lens extra galaxy light is modeled using light profiles [no prior initialization].
     - The lens extra galaxy mass is modeled using a total mass distribution [no prior initialization].
     - The source galaxy's light is a light profiles [no prior initialization].
     - The main lens galaxy is fixed from SOURCE LP 1 [instance]

    This search aims to accurately estimate an initial lens light model, mass model and source model.
    """


    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                mass=lens_mass,
                shear=shear,
            ),
            source_1=af.Model(
                al.Galaxy,
                redshift=redshift_source_1,
                bulge=source_bulge_1,
                mass=source_mass_1,
            ),
            source_2=af.Model(
                al.Galaxy,
                redshift=redshift_source_2,
                bulge=source_bulge_2,
            ),
        ),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="source_lp[2]",
        **settings_search.search_dict,
        n_live=150,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result

def run_cosmo(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    lens_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    lens_mass: af.Model = af.Model(al.mp.PowerLaw),
    shear: af.Model(al.mp.ExternalShear) = af.Model(al.mp.ExternalShear),
    source_bulge_1: Optional[af.Model] = af.Model(al.lp.Sersic),
    source_mass_1: Optional[af.Model] = af.Model(al.mp.Isothermal),
    source_bulge_2: Optional[af.Model] = af.Model(al.lp.Sersic),
    redshift_lens: float = 0.5,
    redshift_source_1: float = 1.0,
    redshift_source_2: float = 2.0,
    cosmology: Optional[af.Model] = af.Model(al.cosmo.FlatwCDMWrap),
    extra_galaxies: Optional[af.Collection] = None,
) -> af.Result:
    """
    The SlaM SOURCE LP PIPELINE, which provides an initial model for the lens's light, mass and source using a
    parametric source model (e.g. MGE, Sersics).

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    lens_bulge
        The model used to represent the light distribution of the lens galaxy's bulge (set to
        None to omit a bulge).
    lens_disk
        The model used to represent the light distribution of the lens galaxy's disk (set to
        None to omit a disk).
    lens_point
        The model used to represent the light distribution of point source emission in the lens galaxy (set to
        None to omit a point).
    mass
        The `MassProfile` fitted by this pipeline.
    shear
        The model used to represent the external shear in the mass model (set to None to turn off shear).
    source_bulge
        The model used to represent the light distribution of the source galaxy's bulge (set to
        None to omit a bulge).
    source_disk
        The model used to represent the light distribution of the source galaxy's disk (set to
        None to omit a disk).
    redshift_lens
        The redshift of the lens galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    redshift_source
        The redshift of the source galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 2 of the SOURCE LP PIPELINE fits a lens model where:

     - The lens extra galaxy light is modeled using light profiles [no prior initialization].
     - The lens extra galaxy mass is modeled using a total mass distribution [no prior initialization].
     - The source galaxy's light is a light profiles [no prior initialization].
     - The main lens galaxy is fixed from SOURCE LP 1 [instance]

    This search aims to accurately estimate an initial lens light model, mass model and source model.
    """

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                mass=lens_mass,
                shear=shear,
            ),
            source_1=af.Model(
                al.Galaxy,
                redshift=redshift_source_1,
                bulge=source_bulge_1,
                mass=source_mass_1,
            ),
            source_2=af.Model(
                al.Galaxy,
                redshift=redshift_source_2,
                bulge=source_bulge_2,
            ),
        ),
        extra_galaxies=extra_galaxies,
        cosmology=cosmology,
    )

    search = af.Nautilus(
        name="cosmology[1]",
        **settings_search.search_dict,
        n_live=150,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result