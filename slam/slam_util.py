from typing import Tuple, Optional, Union

import autofit as af
import autolens as al



def pass_light_and_mass_profile_priors(
    lmp_model: af.Model(al.lmp.LightMassProfile),
    result_light_component: af.Model,
    result: af.Result,
    einstein_mass_range: Optional[Tuple[float, float]] = None,
    as_instance: bool = False,
) -> Optional[af.Model]:
    """
    Returns an updated version of a `LightMassProfile` model (e.g. a bulge or disk) whose priors are initialized from
    previous results of a `Light` pipeline.

    This function generically links any `LightProfile` to any `LightMassProfile`, pairing parameters which share the
    same path.

    It also allows for an Einstein mass range to be input, such that the `LogUniformPrior` on the mass-to-light
    ratio of the lmp_model-component is set with lower and upper limits that are a multiple of the Einstein mass
    computed in the previous SOURCE PIPELINE. For example, if `einstein_mass_range=[0.01, 5.0]` the mass to light
    ratio will use priors corresponding to values which give Einstein masses 1% and 500% of the estimated Einstein mass.

    Parameters
    ----------
    lmp_model : af.Model(al.lmp.LightMassProfile)
        The light and mass profile whoses priors are passed from the LIGHT PIPELINE.
    result_light_component : af.Result
        The `LightProfile` result of the LIGHT PIPELINE used to pass the priors.
    result : af.Result
        The result of the LIGHT PIPELINE used to pass the priors.
    einstein_mass_range : (float, float)
        The values a the estimate of the Einstein Mass in the LIGHT PIPELINE is multiplied by to set the lower and
        upper limits of the profile's mass-to-light ratio.
    as_instance
        If `True` the prior is set up as an instance, else it is set up as a lmp_model component.

    Returns
    -------
    af.Model(mp.LightMassProfile)
        The light and mass profile whose priors are initialized from a previous result.
    """

    if lmp_model is None:
        return lmp_model

    lmp_model.take_attributes(source=result_light_component)

    return lmp_model



def source__from(result: af.Result, source_is_model: bool = False) -> af.Model:
    """
    Setup the source model using the previous pipeline and search results.

    The source light model is not specified by the  MASS PIPELINE and the previous SOURCE PIPELINE is used to
    determine whether the source model is parametric or an inversion.

    The source can be returned as an `instance` or `model`, depending on the optional input. The default SLaM
    pipelines return parametric sources as a model (give they must be updated to properly compute a new mass
    model) and return inversions as an instance (as they have sufficient flexibility to typically not required
    updating). They use the *source_from_pevious_pipeline* method of the SLaM class to do this.

    Parameters
    ----------
    result : af.Result
        The result of the previous source pipeline.
    setup_adapt
        The setup of the adapt fit.
    source_is_model
        If `True` the source is returned as a *model* where the parameters are fitted for using priors of the
        search result it is loaded from. If `False`, it is an instance of that search's result.
    """

    if not hasattr(result.instance.galaxies.source, "pixelization"):

        if source_is_model:

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                bulge=result.model.galaxies.source.bulge,
                disk=result.model.galaxies.source.disk,
            )

        else:

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                bulge=result.instance.galaxies.source.bulge,
                disk=result.instance.galaxies.source.disk,
            )

    if hasattr(result, "adapt"):

        if source_is_model:

            pixelization = af.Model(
                al.Pixelization,
                mesh=result.adapt.instance.galaxies.source.pixelization.mesh,
                regularization=result.adapt.model.galaxies.source.pixelization.regularization,
            )

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                pixelization=pixelization,
            )

        else:

            pixelization = af.Model(
                al.Pixelization,
                mesh=result.adapt.instance.galaxies.source.pixelization.mesh,
                regularization=result.adapt.instance.galaxies.source.pixelization.regularization,
            )

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                pixelization=pixelization,
            )

    else:

        if source_is_model:

            pixelization = af.Model(
                al.Pixelization,
                mesh=result.instance.galaxies.source.pixelization.mesh,
                regularization=result.model.galaxies.source.pixelization.regularization,
            )

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                pixelization=pixelization,
            )

        else:

            pixelization = af.Model(
                al.Pixelization,
                mesh=result.instance.galaxies.source.pixelization.mesh,
                regularization=result.instance.galaxies.source.pixelization.regularization,
            )

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                pixelization=pixelization,
            )


def source__from_result_model_if_parametric(
    result: af.Result,
) -> af.Model:
    """
    Setup the source model for a MASS PIPELINE using the previous SOURCE PIPELINE results.

    The source light model is not specified by the  MASS PIPELINE and the previous SOURCE PIPELINE is used to
    determine whether the source model is parametric or an inversion.

    The source is returned as a model if it is parametric (given its parameters must be fitted for to properly compute
    a new mass model) whereas inversions are returned as an instance (as they have sufficient flexibility to not
    require updating). This behaviour can be customized in SLaM pipelines by replacing this method with the
    `source__from` method.

    Parameters
    ----------
    result
        The result of the previous source pipeline.
    setup_adapt
        The setup of the adapt fit.
    """

    # TODO : Should not depend on name of pixelization being "pixelization"

    if hasattr(result.instance.galaxies.source, "pixelization"):
        if result.instance.galaxies.source.pixelization is not None:
            return source__from(result=result, source_is_model=False)
    return source__from(result=result, source_is_model=True)


def clean_clumps_of_adapt_images(clumps):

    for clump in clumps:

        if hasattr(clump, "adapt_model_image"):
            del clump.adapt_model_image

        if hasattr(clump, "adapt_galaxy_image"):
            del clump.adapt_galaxy_image


def clumps_from(
    result: af.Result, light_as_model: bool = False, mass_as_model: bool = False
):

    # ideal API:

    # clumps = result.instance.clumps.as_model((al.LightProfile, al.mp.MassProfile,), fixed="centre", prior_pass=True)

    if mass_as_model:

        clumps = result.instance.clumps.as_model((al.mp.MassProfile,))

        for clump_index in range(len(result.instance.clumps)):

            if hasattr(result.instance.clumps[clump_index], "mass"):
                clumps[clump_index].mass.centre = result.instance.clumps[
                    clump_index
                ].mass.centre
                clumps[clump_index].mass.einstein_radius = result.model.clumps[
                    clump_index
                ].mass.einstein_radius

    elif light_as_model:

        clumps = result.instance.clumps.as_model((al.LightProfile,))

        for clump_index in range(len(result.instance.clumps)):

            if clumps[clump_index].light is not None:

                clumps[clump_index].light.centre = result.instance.clumps[
                clump_index
            ].light.centre
    #     clumps[clump_index].light.intensity = result.model.clumps[clump_index].light.intensity
    #     clumps[clump_index].light.effective_radius = result.model.clumps[clump_index].light.effective_radius
    #     clumps[clump_index].light.sersic_index = result.model.clumps[clump_index].light.sersic_index

    else:

        clumps = result.instance.clumps.as_model(())

    clean_clumps_of_adapt_images(clumps=clumps)

    return clumps


# TODO : Think about how Rich can full generize these.


def lp_from(
    component: Union[al.LightProfile], fit: Union[al.FitImaging, al.FitInterferometer]
) -> al.LightProfile:

    if isinstance(component, al.lp_linear.LightProfileLinear):

        intensity = fit.linear_light_profile_intensity_dict[component]

        return component.lp_instance_from(intensity=intensity)

    elif isinstance(component, al.lp_basis.Basis):

        light_profile_list = []

        for light_profile in component.light_profile_list:

            intensity = fit.linear_light_profile_intensity_dict[light_profile]

            if isinstance(light_profile, al.lp_linear.LightProfileLinear):

                light_profile_list.append(
                    light_profile.lp_instance_from(intensity=intensity)
                )

            else:

                light_profile_list.append(light_profile)

        #   basis = af.Model(al.lp_basis.Basis, light_profile_list=light_profile_list)

        basis = al.lp_basis.Basis(light_profile_list=light_profile_list)

        return basis

    return component


def lmp_from(
    lp: Union[al.LightProfile, al.lp_linear.LightProfileLinear],
    fit: Union[al.FitImaging, al.FitInterferometer],
) -> al.lmp.LightMassProfile:

    if isinstance(lp, al.lp_linear.LightProfileLinear):

        intensity = fit.linear_light_profile_intensity_dict[lp]

        return lp.lmp_model_from(intensity=intensity)

    return lp
