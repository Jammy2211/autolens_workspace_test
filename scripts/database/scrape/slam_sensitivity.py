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
dataset_name = "no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.2,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
)

dataset = dataset.apply_mask(mask=mask)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=path.join("database", "scrape", "slam_sensitivity"),
    number_of_cores=2,
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
__SOURCE LP PIPELINE (no lens light)__

The SOURCE LP PIPELINE (no lens light) uses one search to initialize a robust model for the source galaxy's 
light, which in this example:

 - Uses a parametric `Sersic` bulge for the source's light (omitting a disk / envelope).
 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.
 - Fixes the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=dataset)

source_results = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=None,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp_linear.Sersic),
    redshift_lens=0.5,
    redshift_source=1.0,
)

"""
__MASS TOTAL PIPELINE (no lens light)__

The MASS TOTAL PIPELINE (no lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIPELINE to initialize the model priors. In this example it:

 - Uses an `PowerLaw` model for the lens's total mass distribution [The centre if unfixed from (0.0, 0.0)].
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS PIPELINE.
"""
analysis = al.AnalysisImaging(dataset=dataset)

mass_results = slam.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_results=source_results,
    light_results=None,
    mass=af.Model(al.mp.PowerLaw),
)

"""
__SUBHALO PIPELINE (sensitivity mapping)__

The SUBHALO PIPELINE (sensitivity mapping) performs sensitivity mapping of the data using the lens model
fitted above, so as to determine where subhalos of what mass could be detected in the data. A full description of
Sensitivity mapping if given in the SLaM pipeline script `slam/subhalo/sensitivity_imaging.py`.

Each model-fit performed by sensitivity mapping creates a new instance of an `Analysis` class, which contains the
data simulated by the `simulate_cls` for that model. This requires us to write a wrapper around the 
PyAutoLens `AnalysisImaging` class.
"""
subhalo_results = slam.subhalo.sensitivity_imaging_lp.run(
    settings_search=settings_search,
    mask=mask,
    psf=dataset.psf,
    mass_results=mass_results,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
)

"""
___Database__

The name of the database, which corresponds to the output results folder.
"""
database_file = "database_directory_slam_sensitivity.sqlite"

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
    directory=path.join("output", "database", "scrape", "slam_sensitivity")
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
    """
    This should return an instance of the `SensitivityResult` object.
    """
    sensitivity_result = fit_grid["result"]

    """
    The log likelihoods of the base fits, perturbed fits and their difference.
    """
    print(sensitivity_result.log_likelihoods_base)
    print(sensitivity_result.log_likelihoods_perturbed)
    print(sensitivity_result.log_likelihood_differences)
