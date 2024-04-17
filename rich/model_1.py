import numpy as np

import autofit as af
import autolens as al

mask_radius = 3.0

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

total_gaussians = 20
gaussian_per_basis = 1

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
    light_profile_list=bulge_gaussian_list,
)


centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

total_gaussians = 20
gaussian_per_basis = 1

log10_sigma_list = np.linspace(-2, np.log10(1.0), total_gaussians)

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
    light_profile_list=bulge_gaussian_list,
)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            bulge=lens_bulge,
        ),
        source=af.Model(
            al.Galaxy,
            bulge=source_bulge,
        ),
    ),
)

print(model.info)

"""
__Things that Need Fixing__

The lens_bulge has 20 guassians (see line 11), however the integer spans on the model.info only go 0 - 9, for example:

galaxies
    lens
        redshift                                                                Prior Missing: Enter Manually or Add to Config
        bulge
            light_profile_list
                0 - 9                        <------------- THIS SHOULD BE 0-19
                    centre
                        centre_0                                                UniformPrior [0], lower_limit = -0.1, upper_limit = 0.1
                        centre_1                                                UniformPrior [1], lower_limit = -0.1, upper_limit = 0.1
                    ell_comps
                        ell_comps_0                                             GaussianPrior [4], mean = 0.0, sigma = 0.3
                        ell_comps_1                                             GaussianPrior [5], mean = 0.0, sigma = 0.3                    
"""