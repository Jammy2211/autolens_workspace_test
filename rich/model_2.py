import numpy as np
from os import path

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
            redshift=0.5,
            bulge=lens_bulge,
            disk=lens_disk,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
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

                1 - 41          <- should be 1, 21, 41
                    sigma                                                       0.013501275609964633


It is true that Gaussian 1 and 41 have the same sigma value (21 also does). However, using 1 - 41 implies that all gaussians
between 1 and 41 have the same sigma value. This is not true.

I think we need a clause where:

- If the range includes all consecutive integers, for example 0, 1, 2, 3, 4, then use 0 - 4
- If it does not, write out the integers individually, for example 0, 21, 41

"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "slam"))


dataset_label = "build"
dataset_type = "imaging"
dataset_name = "with_lens_light"

dataset_path = path.join("dataset", dataset_label, dataset_type, dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)


mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.8,
    outer_radius=2.6,
)

dataset = dataset.apply_mask(mask=mask)

analysis = al.AnalysisImaging(
    dataset=dataset,
)

search = af.DynestyStatic(
    path_prefix=path.join("model_2"),
    nlive=50,
    maxcall=10,
    maxiter=10,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

print(result.info)

"""
__Things that Need Fixing__

The results are also listed as 1 - 41, when they should instead be numbers:

                1 - 41   <- Should be 1, 21, 41
                    sigma                                                       0.013501275609964633
                    
Ranges are lost on many results:

            light_profile_list
                59              <- should be 0 - 59
                    centre
                        centre_0                                                0.0612 (-0.0980, 0.0972)
                        centre_1                                                -0.0110 (-0.0949, 0.0916)
                    ell_comps
                        ell_comps_0                                             -0.2109 (-0.7167, 0.7622)
                        ell_comps_1                                             0.0756 (-0.6566, 0.6295)
                19              <- should be 0 - 19
                    ell_comps
                        ell_comps_0                                             -0.0142 (-0.6537, 0.9201)
                        ell_comps_1                                             0.0705 (-0.4976, 0.8441)
                39              <- should be 19 - 39
                    ell_comps
                        ell_comps_0                                             -0.0698 (-0.8218, 0.5944)
                        ell_comps_1                                             -0.0889 (-0.6299, 0.5197)
"""