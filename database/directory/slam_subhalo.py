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

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=path.join("database", "directory", "slam_subhalo"),
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
# disk = af.Model(al.lp.Sersic)
bulge.centre = disk.centre

source_lp_result = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=bulge,
    lens_disk=disk,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.Sersic),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
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
bulge = af.Model(al.lp.Sersic)
disk = af.Model(al.lp.Exponential)
bulge.centre = disk.centre

light_results = slam.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result=source_lp_result,
    lens_bulge=bulge,
    lens_disk=disk,
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

multipole = af.Model(al.mp.PowerLawMultipole)
multipole.m = 4

mass_results = slam.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_results=source_lp_result,
    light_result=light_results,
    mass=af.Model(al.mp.PowerLaw),
    multipole=multipole,
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

subhalo_results = slam.subhalo.detection.run(
    settings_search=settings_search,
    analysis=analysis,
    mass_results=mass_results,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
)

"""
___Database__

The name of the database, which corresponds to the output results folder.
"""
database_file = "database_directory_slam_subhalo.sqlite"

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
    directory=path.join("output", "database", "directory", "slam_subhalo")
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
fit_imaging_gen = fit_imaging_agg.max_log_likelihood_gen_from()

info_gen = agg_best_fits.values("info")

for fit_grid, fit_imaging_detect, info in zip(agg_grid, fit_imaging_gen, info_gen):
    grid_search_result = fit_grid["result"]

    """
    The log likelihoods of the grid search result, on a native 2D grid.
    """
    print(grid_search_result.log_likelihoods_native)

    # subhalo_search_result = al.subhalo.SubhaloGridSearchResult(
    #     grid_search_result=grid_search_result, result_no_subhalo=fit_grid.parent
    # )
    #
    # plot_path = path.join("database", "plot", "slam_subhalo", "likelihood")
    #
    # mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png"))
    #
    # subhalo_plotter = al.subhalo.SubhaloPlotter(
    #     subhalo_result=subhalo_search_result,
    #     fit_imaging_detect=fit_imaging_detect,
    #     use_log_evidences=False,
    #     use_stochastic_log_evidences=False,
    #     mat_plot_2d=mat_plot_2d,
    # )
    # subhalo_plotter.subplot_detection_imaging(remove_zeros=True)
    # subhalo_plotter.subplot_detection_fits()
    # subhalo_plotter.set_filename(filename="image_2d")
    # subhalo_plotter.figure_figures_of_merit_grid(image=True, remove_zeros=False)
    # subhalo_plotter.set_filename(filename="image_2d")
    # subhalo_plotter.figure_figures_of_merit_grid(image=True, remove_zeros=True)
    #
    # plot_path = path.join("database", "plot", "slam_subhalo", "evidence")
    #
    # mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png"))
    #
    # subhalo_plotter = al.subhalo.SubhaloPlotter(
    #     subhalo_result=subhalo_search_result,
    #     fit_imaging_detect=fit_imaging_detect,
    #     use_log_evidences=True,
    #     use_stochastic_log_evidences=False,
    #     mat_plot_2d=mat_plot_2d,
    # )
    # subhalo_plotter.subplot_detection_imaging(remove_zeros=True)
    # subhalo_plotter.subplot_detection_fits()
    # subhalo_plotter.set_filename(filename="image_2d")
    # subhalo_plotter.figure_figures_of_merit_grid(image=True, remove_zeros=False)
    # subhalo_plotter.set_filename(filename="image_2d")
    # subhalo_plotter.figure_figures_of_merit_grid(image=True, remove_zeros=True)
    #
    # try:
    #
    #     plot_path = path.join("database", "plot", "slam_subhalo", "stochastic")
    #
    #     mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png"))
    #
    #     subhalo_plotter = al.subhalo.SubhaloPlotter(
    #         subhalo_result=subhalo_search_result,
    #         fit_imaging_detect=fit_imaging_detect,
    #         use_log_evidences=True,
    #         use_stochastic_log_evidences=True,
    #         mat_plot_2d=mat_plot_2d,
    #     )
    #     subhalo_plotter.subplot_detection_imaging(remove_zeros=True)
    #     subhalo_plotter.subplot_detection_fits()
    #     subhalo_plotter.set_filename(filename="image_2d")
    #     subhalo_plotter.figure_figures_of_merit_grid(image=True, remove_zeros=False)
    #     subhalo_plotter.set_filename(filename="image_2d")
    #     subhalo_plotter.figure_figures_of_merit_grid(image=True, remove_zeros=True)
    #
    # except ValueError:
    #
    #     pass
