"""
Database: Model-Fit
===================

This is a simple example of a model-fit which we wish to write to the database. This should simply output the
results to the `.sqlite` database file.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import os
from os import path

cwd = os.getcwd()
from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "fit"))


import autofit as af
import autolens as al
import autolens.plot as aplt
import slam

"""
__Dataset + Masking__
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.2,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_autofit = af.SettingsSearch(
    path_prefix=path.join("database", "directory", "subhalo_slam"),
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
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit as is used identically to the
hyper pipeline examples.

The `SetupHyper` input `hyper_fixed_after_source` fixes the hyper-parameters to the values computed by the hyper 
extension at the end of the SOURCE PIPELINE. By fixing the hyper-parameter values at this point, model comparison 
of different models in the LIGHT PIPELINE and MASS PIPELINE can be performed consistently.
"""
setup_hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
)

"""
__SOURCE PARAMETRIC PIPELINE (no lens light)__

The SOURCE PARAMETRIC PIPELINE (no lens light) uses one search to initialize a robust model for the source galaxy's 
light, which in this example:

 - Uses a parametric `EllSersic` bulge for the source's light (omitting a disk / envelope).
 - Uses an `EllIsothermal` model for the lens's total mass distribution with an `ExternalShear`.
 - Fixes the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=imaging)

source_results = slam.source_parametric.no_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    mass=af.Model(al.mp.EllIsothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.EllSersic),
    redshift_lens=0.5,
    redshift_source=1.0,
)

"""
__MASS TOTAL PIPELINE (no lens light)__

The MASS TOTAL PIPELINE (no lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIPELINE to initialize the model priors. In this example it:

 - Uses an `EllPowerLaw` model for the lens's total mass distribution [The centre if unfixed from (0.0, 0.0)].
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS PIPELINE.
"""
analysis = al.AnalysisImaging(dataset=imaging)

mass_results = slam.mass_total.no_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_results,
    mass=af.Model(al.mp.EllPowerLaw),
)

"""
__SUBHALO PIPELINE (single plane detection)__

The SUBHALO PIPELINE (single plane detection) consists of the following searches:

 1) Refit the lens and source model, to refine the model evidence for comparing to the models fitted which include a 
 subhalo. This uses the same model as fitted in the MASS PIPELINE. 
 2) Performs a grid-search of non-linear searches to attempt to detect a dark matter subhalo. 
 3) If there is a successful detection a final search is performed to refine its parameters.

For this runner the `SetupSubhalo` customizes:

 - If the parameteric source galaxy is treated as a model (all free parameters) or instance (all fixed) during the 
   subhalo detection grid search.
 - The NxN size of the grid-search.
"""
analysis = al.AnalysisImaging(dataset=imaging)

subhalo_results = slam.subhalo.detection_single_plane(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    mass_results=mass_results,
    subhalo_mass=af.Model(al.mp.SphNFWMCRLudlow),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
)

"""
___Database__

The name of the database, which corresponds to the output results folder.
"""
database_file = "database_directory_subhalo_slam.sqlite"

"""
Remove database is making a new build (you could delete manually via your mouse). Building the database is slow, so 
only do this when you redownload results. Things are fast working from an already built database.
"""
try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass

"""
Load the database. If the file `slacs.sqlite` does not exist, it will be made by the method below, so its fine if
you run the code below before the file exists.
"""
agg = af.Aggregator.from_database(filename=database_file, completed_only=False)

"""
Add all results in the directory "output/slacs" to the database, which we manipulate below via the agg.
Avoid rerunning this once the file `slacs.sqlite` has been built.
"""
agg.add_directory(
    directory=path.join("output", "database", "directory", "subhalo_slam")
)

"""
__Query__
"""

agg_grid = agg.grid_searches()

"""
Unique Tag Query Does Not Work
"""
agg_best_fits = agg_grid.best_fits()

fit_imaging_agg = al.agg.FitImagingAgg(aggregator=agg_best_fits)
fit_imaging_gen = fit_imaging_agg.max_log_likelihood_gen()

info_gen = agg_best_fits.values("info")

for fit_grid, fit_imaging_detect, info in zip(agg_grid, fit_imaging_gen, info_gen):

    subhalo_search_result = al.subhalo.SubhaloResult(
        grid_search_result=fit_grid["result"], result_no_subhalo=fit_grid.parent
    )

    plot_path = path.join("database", "plot", "subhalo_slam", "likelihood")

    mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png"))

    subhalo_plotter = al.subhalo.SubhaloPlotter(
        subhalo_result=subhalo_search_result,
        fit_imaging_detect=fit_imaging_detect,
        use_log_evidences=False,
        use_stochastic_log_evidences=False,
        mat_plot_2d=mat_plot_2d,
    )
    subhalo_plotter.subplot_detection_imaging(remove_zeros=True)
    subhalo_plotter.subplot_detection_fits()
    subhalo_plotter.set_filename(filename="image_2d")
    subhalo_plotter.figure_with_detection_overlay(image=True, remove_zeros=False)
    subhalo_plotter.set_filename(filename="image_2d")
    subhalo_plotter.figure_with_detection_overlay(image=True, remove_zeros=True)

    plot_path = path.join("database", "plot", "subhalo_slam", "evidence")

    mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png"))

    subhalo_plotter = al.subhalo.SubhaloPlotter(
        subhalo_result=subhalo_search_result,
        fit_imaging_detect=fit_imaging_detect,
        use_log_evidences=True,
        use_stochastic_log_evidences=False,
        mat_plot_2d=mat_plot_2d,
    )
    subhalo_plotter.subplot_detection_imaging(remove_zeros=True)
    subhalo_plotter.subplot_detection_fits()
    subhalo_plotter.set_filename(filename="image_2d")
    subhalo_plotter.figure_with_detection_overlay(image=True, remove_zeros=False)
    subhalo_plotter.set_filename(filename="image_2d")
    subhalo_plotter.figure_with_detection_overlay(image=True, remove_zeros=True)

    try:

        plot_path = path.join("database", "plot", "subhalo_slam", "stochastic")

        mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png"))

        subhalo_plotter = al.subhalo.SubhaloPlotter(
            subhalo_result=subhalo_search_result,
            fit_imaging_detect=fit_imaging_detect,
            use_log_evidences=True,
            use_stochastic_log_evidences=True,
            mat_plot_2d=mat_plot_2d,
        )
        subhalo_plotter.subplot_detection_imaging(remove_zeros=True)
        subhalo_plotter.subplot_detection_fits()
        subhalo_plotter.set_filename(filename="image_2d")
        subhalo_plotter.figure_with_detection_overlay(image=True, remove_zeros=False)
        subhalo_plotter.set_filename(filename="image_2d")
        subhalo_plotter.figure_with_detection_overlay(image=True, remove_zeros=True)

    except ValueError:

        pass
