import numpy as np
from os import path

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
    profile_list=bulge_gaussian_list,
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
    profile_list=bulge_gaussian_list,
)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=lens_bulge,
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
    path_prefix=path.join("model_1"),
    nlive=50,
    maxcall=10,
    maxiter=10,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

print(result.info)

"""
__Things that Need Fixing__

The result is closer to correct, however the indexes do not have spanning and only state the index of the last component:

galaxies
    lens
        bulge
            light_profile_list
                19                            <- THIS SHOULD BE 0 - 19
                    centre
                        centre_0                                                0.0556 (-0.0939, 0.0985)
                        centre_1                                                0.0720 (-0.0962, 0.0958)
                    ell_comps
                        ell_comps_0                                             -0.0194 (-0.9471, 0.8800)
                        ell_comps_1                                             0.0605 (-0.6673, 0.7894)
    source
        bulge
            light_profile_list
                19                            <- THIS SHOULD BE 0 - 19
                    centre
                        centre_0                                                -0.2999 (-0.8971, 0.6269)
                        centre_1                                                0.1025 (-0.7427, 0.6934)
                    ell_comps
                        ell_comps_0                                             -0.0441 (-0.6477, 0.6917)
                        ell_comps_1                                             0.4576 (-0.9851, 0.6987)
"""
