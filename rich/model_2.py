import numpy as np

import autofit as af
import autolens as al

mask_radius = 3.0

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

total_gaussians = 20
gaussian_per_basis = 3

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

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

disk_gaussian_list = []

for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]

    disk_gaussian_list += gaussian_list

lens_disk = af.Model(
    al.lp_basis.Basis,
    light_profile_list=disk_gaussian_list,
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
            disk=lens_disk,
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

The split of bulge - disk and disk does not make any sense:

model                                                                           Collection (N=18)
    galaxies                                                                    Collection (N=18)
        lens                                                                    Galaxy (N=14)
            bulge - disk                                                        Basis (N=8)
                light_profile_list                                              Collection (N=8)
                    20 - 59                                                     Gaussian (N=4)
            disk
                light_profile_list
                    0 - 9                                                       Gaussian (N=4)
                    
The bulge has 60 unique gaussians, as does the disk. So I think this should read:

model                                                                           Collection (N=18)
    galaxies                                                                    Collection (N=18)
        lens                                                                    Galaxy (N=14)
            bulge - disk                                                        Basis (N=8)
                light_profile_list                                              Collection (N=8)
                    0 - 59                                                      Gaussian (N=4)
                    
I guess this is happening because of how the centre prior is only tied to Gaussians 0-19 and shared across 20-39 and 40-59.



Assining the centre to only gaussians 0 - 9 is strange:

        bulge - disk
            light_profile_list
                0 - 9
                    centre
                        centre_0                                                UniformPrior [0], lower_limit = -0.1, upper_limit = 0.1
                        centre_1                                                UniformPrior [1], lower_limit = -0.1, upper_limit = 0.1
                        
Gaussians 10-59 also share this prior via the code:

        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        
I guess this should be 0 - 59.



The following shows a limitation of using the # - # API:

                1 - 41
                    sigma                                                       0.013501275609964633


It is true that Gaussian 1 and 41 have the same sigma value (21 also does). However, using 1 - 41 implies that all gaussians
between 1 and 41 have the same sigma value. This is not true.

I think we need a clause where:

- If the range includes all consecutive integers, for example 0, 1, 2, 3, 4, then use 0 - 4
- If it does not, write out the integers individually, for example 0, 21, 41

"""