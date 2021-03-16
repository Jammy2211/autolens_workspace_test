"""
Pipelines: Light Parametric + Mass Total + Source Parametric
============================================================

By chaining together five searches this script fits strong lens `Imaging`, where in the final model:

 - The lens galaxy's light is a parametric `EllipticalSersic`.
 - The lens galaxy's total mass distribution is an `EllipticalIsothermal`.
 - The source galaxy is a parametric `EllipticalSersic`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__ 

Load the `Imaging` data, define the `Mask2D` and plot them.
"""
dataset_name = "light_sersic__mass_sie__source_sersic"
pixel_scales = 0.1

dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scales,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
__Settings__

The `SettingsPhaseImaging` describe how the model is fitted to the data in the log likelihood function.

These settings are used and described throughout the `autolens_workspace/notebooks/imaging/modeling` example scripts, with a 
complete description of all settings given in `autolens_workspace/notebooks/imaging/modeling/customize/settings.py`.

The settings chosen here are applied to all phases in the pipeline.
"""
settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid2D, sub_size=2)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

"""
__Pipeline_Setup__:

Pipelines use `Setup` objects to customize how different aspects of the model are fitted. 

First, we create a `SetupLightParametric` which customizes:

 - The `LightProfile`'s which fit different components of the lens light, such as its `bulge` and `disk`.
 - The alignment of these components, for example if the `bulge` and `disk` centres are aligned.
 - If the centre of the lens light profile is manually input and fixed for modeling.
 
In this example we fit the lens light as just one component, a `bulge` represented as an `EllipticalSersic`. We do 
not fix its centre to an input value. We have included options of `SetupLightParametric` with input values of
`None`, illustrating how it could be edited to fit different models.
"""
setup_light = al.SetupLightParametric(
    bulge_prior_model=al.lp.EllipticalSersic,
    disk_prior_model=None,
    envelope_prior_model=None,
    align_bulge_disk_centre=False,
    align_bulge_disk_elliptical_comps=False,
    light_centre=None,
)

"""
This pipeline also uses a `SetupMass`, which customizes:

 - The `MassProfile` used to fit the lens's total mass distribution.
 - If there is an `ExternalShear` in the mass model or not.
  - If the centre of the `MassProfile` is aligned with the centre of the `LightProfile`'s.
"""
setup_mass = al.SetupMassTotal(
    mass_prior_model=al.mp.EllipticalPowerLaw,
    with_shear=True,
    align_bulge_mass_centre=False,
)

"""
Next, we create a `SetupSourceParametric` which customizes the parametric source model in an identical way to how 
`SetupLightParametric` customizes the parametric lens light model.

This example uses a single `bulge` with an `EllipticalSersic` and we've included examples illustrating how this could 
be customized.
"""
setup_source = al.SetupSourceParametric(
    bulge_prior_model=al.lp.EllipticalSersic,
    disk_prior_model=None,
    envelope_prior_model=None,
    align_bulge_disk_centre=False,
    align_bulge_disk_elliptical_comps=False,
    light_centre=None,
)

"""
_Pipeline Tagging_

The `Setup` objects are input into a `SetupPipeline` object, which is passed into the pipeline and used to customize
the analysis depending on the setup. This includes tagging the output path of a pipeline. For example, if `with_shear` 
is True, the pipeline`s output paths are `tagged` with the string `with_shear`.

This means you can run the same pipeline on the same data twice (e.g. with and without shear) and the results will go
to different output folders and thus not clash with one another!

The `path_prefix` below specifies the path the pipeline results are written to, which is:

 `autolens_workspace/output/imaging/modeling/pipelines/with_lens_light/dataset_type/dataset_name` 
 `autolens_workspace/output/imaging/modeling/pipelines/with_lens_light/light_sersic__mass_sie__source_sersic/`

The redshift of the lens and source galaxies are also input (see `notebooks/imaging/modeling/customize/redshift.py`) for a 
description of what inputting redshifts into **PyAutoLens** does.
"""
setup = al.SetupPipeline(
    path_prefix=path.join("imaging", "pipelines", "with_lens_light", dataset_name),
    redshift_lens=0.5,
    redshift_source=1.0,
    setup_light=setup_light,
    setup_mass=setup_mass,
    setup_source=setup_source,
)

"""
__Pipeline Creation__

To create a pipeline we import it from the pipelines folder and run its `make_pipeline` function, inputting the 
`Setup` and `SettingsPhase` above.
"""
from pipelines import light_parametric__mass_total__source_parametric

pipeline = light_parametric__mass_total__source_parametric.make_pipeline(
    setup=setup, settings=settings
)

"""
__Pipeline Run__

Running a pipeline is the same as running a phase, we simply pass it our lens dataset and mask to its run function.
"""
pipeline.run(dataset=imaging, mask=mask)

"""
Finish.
"""