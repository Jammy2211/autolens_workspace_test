"""
SLaM (Source, Light and Mass): Source Light Pixelized + Light Profile + Mass Total + Subhalo NFW
================================================================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS PIPELINE.

Using a SOURCE LP PIPELINE, LIGHT LP PIPELINE, MASS PIPELINE and SUBHALO PIPELINE this SLaM script
fits `Imaging` of a strong lens system, where in the final model:

 - The lens galaxy's light is a bulge+disk `Sersic` and `Exponential`.
 - The lens galaxy's total mass distribution is an `Isothermal`.
 - A dark matter subhalo near The lens galaxy mass is included as a`NFWMCRLudlowSph`.
 - The source galaxy is an `Sersic`.

This uses the SLaM pipelines:

 `source_lp`
 `light_lp`
 `mass_total`
 `subhalo/detection`

Check them out for a full description of the analysis!
"""
from autolens.lens.tracer_util import plane_redshifts_from


def fit():
    # %matplotlib inline
    # from pyprojroot import here
    # workspace_path = str(here())
    # %cd $workspace_path
    # print(f"Working Directory has been set to `{workspace_path}`")

    import os
    from os import path

    import autofit as af
    import autolens as al
    import autolens.plot as aplt
    import slam

    """
    __Dataset__ 
    
    Load the `Imaging` data, define the `Mask2D` and plot them.
    """
    dataset_name = "with_lens_light"
    dataset_path = path.join("dataset", "imaging", dataset_name)

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        pixel_scales=0.2,
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    dataset = dataset.apply_mask(mask=mask)

    dataset_plotter = aplt.ImagingPlotter(
        dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
    )
    dataset_plotter.subplot_dataset()

    """
    __Settings AutoFit__
    
    The settings of autofit, which controls the output paths, parallelization, database use, etc.
    """
    settings_search = af.SettingsSearch(
        path_prefix=path.join("slam", "source_lp", "mass_total", "base"),
        unique_tag="hi",
        number_of_cores=1,
        session=None,
    )

    """
    __Redshifts__
    
    The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
    from arc-seconds to kiloparsecs, masses to solar masses, etc.).
    """
    redshift_lens = 0.5
    redshift_source = 1.0

    """
    __SOURCE LP PIPELINE (with lens light)__
    
    The SOURCE LP PIPELINE (with lens light) uses three searches to initialize a robust model for the 
    source galaxy's light, which in this example:
     
     - Uses a parametric `Sersic` bulge and `Exponential` disk with centres aligned for the lens
     galaxy's light.
     
     - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.
    
     Settings:
    
     - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS PIPELINE).
    """
    analysis = al.AnalysisImaging(dataset=dataset)

    bulge = af.Model(al.lp.Sersic)
    disk = af.Model(al.lp.Exponential)
    bulge.centre = disk.centre

    source_lp_result = slam.source_lp.run_DSPL(
        settings_search=settings_search,
        analysis=analysis,
        lens_bulge=bulge,
      #  lens_disk=disk,
        lens_mass=af.Model(al.mp.Isothermal),
        shear=af.Model(al.mp.ExternalShear),
        source_bulge_1=af.Model(al.lp.Sersic),
      #  redshift_lens=redshift_lens,
      #  redshift_source_1=redshift_source,
    )

    """
    __LIGHT LP PIPELINE__
    
    The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
    lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE LP PIPELINE.
    In this example it:
    
     - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
     light [Do not use the results of the SOURCE LP PIPELINE to initialize priors].
    
     - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE LP PIPELINE].
    
     - Uses the `Sersic` model representing a bulge for the source's light [fixed from SOURCE LP PIPELINE].
    
     - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
     PIPELINE [fixed values].
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2, plane_redshift=source_lp_result.instance.galaxies.source_1.redshift,
            ),
            source_lp_result.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2, plane_redshift=source_lp_result.instance.galaxies.source_2.redshift,
            )
        ],
    )

    print(analysis.positions_likelihood_list[0].positions)
    print(analysis.positions_likelihood_list[1].positions)

    bulge = af.Model(al.lp.Sersic)
    disk = af.Model(al.lp.Exponential)
    bulge.centre = disk.centre

    light_result = slam.light_lp.run_DSPL(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        lens_bulge=bulge,
#        lens_disk=disk,
    )

    """
    __MASS TOTAL PIPELINE (with lens light)__
    
    The MASS TOTAL PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
    using the lens mass model and source model of the SOURCE PIPELINE to initialize the model priors and the lens light
    model of the LIGHT LP PIPELINE. In this example it:
    
     - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
     light [fixed from LIGHT LP PIPELINE].
    
     - Uses an `PowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE 
     PARAMETRIC PIPELINE + centre unfixed from (0.0, 0.0)].
     
     - Uses the `Sersic` model representing a bulge for the source's light [priors initialized from SOURCE 
     PARAMETRIC PIPELINE].
     
     - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS PIPELINE.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_lp_result)
    )

    mass_result = slam.mass_total.run(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_lp_result,
        source_result_for_source=source_lp_result,
        light_result=light_result,
        mass=af.Model(al.mp.PowerLaw),
        multipole_4=af.Model(al.mp.PowerLawMultipole),
        reset_shear_prior=True,
    )

    """
    __SUBHALO PIPELINE (single plane detection)__
    
    The SUBHALO PIPELINE (single plane detection) consists of the following searches:
     
     1) Refit the lens and source model, to refine the model evidence for comparing to the models fitted which include a 
     subhalo. This uses the same model as fitted in the MASS PIPELINE. 
     2) Performs a grid-search of non-linear searches to attempt to detect a dark matter subhalo. 
     3) If there is a successful detection a final search is performed to refine its parameters.
     
    For this runner the SUBHALO PIPELINE customizes:
    
     - The [number_of_steps x number_of_steps] size of the grid-search, as well as the dimensions it spans in arc-seconds.
     - The `number_of_cores` used for the gridsearch, where `number_of_cores > 1` performs the model-fits in paralle using
     the Python multiprocessing module.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_lp_result)
    )

    subhalo_result_1 = slam.subhalo.detection.run_1_no_subhalo(
        settings_search=settings_search,
        analysis=analysis,
        mass_result=mass_result,
    )

    subhalo_grid_search_result_2 = slam.subhalo.detection.run_2_grid_search(
        settings_search=settings_search,
        analysis=analysis,
        mass_result=mass_result,
        subhalo_result_1=subhalo_result_1,
        subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
        grid_dimension_arcsec=3.0,
        number_of_steps=2,
    )

    subhalo_result_3 = slam.subhalo.detection.run_3_subhalo(
        settings_search=settings_search,
        analysis=analysis,
        subhalo_result_1=subhalo_result_1,
        subhalo_grid_search_result_2=subhalo_grid_search_result_2,
        subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    )


if __name__ == "__main__":
    fit()
