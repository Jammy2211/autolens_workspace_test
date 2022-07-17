"""
SLaM (Source, Light and Mass): Mass Total + Source Inversion
============================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `EllSersic` profile for the bulge, this will be used in the subsequent MASS PIPELINE.

Using a SOURCE PARAMETRIC PIPELINE, INVERSION SOURCE PIPELINE and a MASS PIPELINE this SLaM script fits `Imaging` of a
strong lens system, where in the final model:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `EllPowerLaw`.
 - The source galaxy is an `Inversion`.

This uses the SLaM pipelines:

 `source__parametric/source_parametric__no_lens_light`
 `source_pixelized/source_pixelized__no_lens_light`
 `mass__total/mass__total__no_lens_light`

Check them out for a detailed description of the analysis!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "slam"))

import autofit as af
import autolens as al
import autolens.plot as aplt
import slam

"""
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
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

imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
___Database__

Build the database.
"""
database_name = "preloads"

try:
    os.remove(
        path.join(
            "output", "slam", "mass_total__source_pixelized", f"{database_name}.sqlite"
        )
    )
except FileNotFoundError:
    pass

agg = af.Aggregator.from_database(
    filename=path.join(
        "output", "slam", "mass_total__source_pixelized", f"{database_name}.sqlite"
    ),
    completed_only=False,
)

agg.add_directory(
    directory=path.join("output", "slam", "mass_total__source_pixelized", database_name)
)

"""
__Query__

Tests that the preloaded sparse image grid is used when making a FitImaging.
"""
name = agg.search.name
agg_query = agg.query(name == "mass_total[1]_mass[total]_source")
samples_gen = agg_query.values("samples")
print(
    "Total Samples Objects via `EllIsothermal` model query = ",
    len(list(samples_gen)),
    "\n",
)

fit_imaging_gen = al.agg.FitImaging(aggregator=agg_query)
fit = list(fit_imaging_gen)

"""
Finish.
"""
