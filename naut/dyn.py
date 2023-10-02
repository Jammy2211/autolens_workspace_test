import numpy as np
from os import path
import sys
import json


def fit():

    """ 
    __AUTOLENS + DATA__
    """
    import autofit as af
    import autolens as al

    pixel_scales = 0.05

    dataset_name = "slacs0946+1006"

    dataset_path = path.join("nautilus_vs_dynesty", dataset_name)

    with open(path.join(dataset_path, "info.json")) as json_file:
        info = json.load(json_file)

    dataset = al.Imaging.from_fits(
        data_path=f"{dataset_path}/image_lens_light_scaled.fits",
        psf_path=f"{dataset_path}/psf.fits",
        noise_map_path=f"{dataset_path}/noise_map_scaled.fits",
        pixel_scales=pixel_scales,
    )

    mask_radius = 3.0

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=pixel_scales,
        centre=(0.0, 0.0),
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)

    dataset = dataset.apply_settings(
        settings=al.SettingsImaging(grid_class=al.Grid2D, sub_size=1)
    )

    """
    __Settings AutoFit__

    The settings of autofit, which controls the output paths, parallelization, databse use, etc.
    """
    number_of_cores = int(sys.argv[1])

    sample_folder = f"dynesty_x{number_of_cores}"

    settings_autofit = af.SettingsSearch(
        path_prefix=path.join(sample_folder),
        unique_tag=dataset_name,
        number_of_cores=number_of_cores,
        session=None,
        info=info,
    )

    """
    __SOURCE LP PIPELINE (with lens light)__

    The SOURCE LP PIPELINE (with lens light) uses three searches to initialize a robust model for the 
    source galaxy's light, which in this example:

     - Uses a parametric `Sersic` bulge and `Exponential` disk with centres aligned for the lens
     galaxy's light.

     - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

     __Settings__:

     - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
    )

    centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    total_gaussians = 20
    gaussian_per_basis = 1

    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):

        gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians))

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

        gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians))

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

        gaussian_list = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians))

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

    model_1 = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=0.5,
                bulge=lens_bulge,
                disk=lens_disk,
                mass=af.Model(al.mp.Isothermal),
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(
                al.Galaxy,
                redshift=1.0,
                bulge=source_bulge
            ),
        ),
    )

    search_1 = af.DynestyStatic(
        name="dyn",
        **settings_autofit.search_dict,
        nlive=200,
        walks=10,
        iterations_per_update=30000
    )

    search_1.fit(
        model=model_1, analysis=analysis, **settings_autofit.fit_dict
    )

if __name__ == "__main__":
    fit()

"""
Finish.
"""
