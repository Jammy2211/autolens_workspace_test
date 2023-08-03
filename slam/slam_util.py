from typing import Tuple, Optional, Union

import autofit as af
import autolens as al


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
