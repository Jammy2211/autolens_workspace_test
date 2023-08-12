
def fit():
    # %matplotlib inline
    # from pyprojroot import here
    # workspace_path = str(here())
    # %cd $workspace_path
    # print(f"Working Directory has been set to `{workspace_path}`")

    import os
    from os import path

    cwd = os.getcwd()

    from autoconf import conf

    conf.instance.push(new_path=path.join(cwd, "config", "searches"))

    import autofit as af
    import autolens as al
    import autolens.plot as aplt

    """
    __Paths__
    """
    dataset_name = "with_lens_light_search"
    path_prefix = path.join("searches", "parametric", "initialization")

    """
    __Search__
    """

    search = af.Nautilus(
        path_prefix=path_prefix,
        name="Nautilus_x8_pool_branch_7_mpi",
        unique_tag=dataset_name,
        n_live=250,
        n_eff=500,
        verbose=True,
        number_of_cores=8,
        mpi=True
    )

    """
    __Dataset + Masking__
    """
    dataset_path = path.join("dataset", "imaging", dataset_name)

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=0.1,
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    dataset = dataset.apply_mask(mask=mask)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    """
    __Model + Search + Analysis + Model-Fit__
    """

    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=al.lp_linear.Sersic, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
    )
    source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.Sersic)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisImaging(dataset=dataset,
                               #  positions_likelihood=al.PositionsLHPenalty(threshold=0.4, positions=positions),
                                  )

    result = search.fit(model=model, analysis=analysis)

if __name__ == "__main__":
    fit()
