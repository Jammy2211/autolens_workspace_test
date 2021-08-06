from os import path
import sys
import json

"""
__COSMA SETUP__
"""
# cosma_path = path.join(path.sep, "cosma5", "data", "durham", "dc-nigh1", "autolens")
# workspace_path = path.join(path.sep, "cosma", "home", "dp004", "dc-nigh1", "slacs")

cosma_path = path.join(
    path.sep, "home", "dc-nigh1", "rds", "rds-dirac-dp195-i2FIP1t5TkY", "dc-nigh1"
)
workspace_path = path.join(path.sep, "home", "dc-nigh1", "profiling")

config_path = path.join(workspace_path, "config")
cosma_dataset_path = path.join(cosma_path, "dataset")
cosma_output_path = path.join(cosma_path, "output")


"""
__Configs__
"""
from autoconf import conf

conf.instance.push(new_path=config_path, output_path=cosma_output_path)

""" 
__AUTOLENS + DATA__
"""
import autofit as af
import autolens as al

import os

sys.path.insert(0, os.getcwd())

instrument = "vro"
# instrument = "euclid"
# instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

dataset_path = path.join(cosma_dataset_path, "imaging", "instruments", instrument)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 4
mask_radius = 3.5
psf_shape_2d = (21, 21)
pixelization_shape_2d = (57, 57)

"""
The model-fit also requires a mask defining the regions of the image we fit the lens model to the data.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    sub_size=sub_size,
    radius=mask_radius,
)


masked_imaging = imaging.apply_mask(mask=mask)
# masked_imaging = masked_imaging.apply_settings(
#     settings=al.SettingsImaging(grid_class=al.Grid2DIterate, grid_inversion_class=al.Grid2D, sub_size_inversion=sub_size)
# )

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=4.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp.EllExponential(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=2.0,
        effective_radius=1.6,
    ),
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=45.0),
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.001, 0.001)),
)

pixelization = al.pix.VoronoiMagnification(shape=pixelization_shape_2d)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
    regularization=al.reg.Constant(coefficient=1.0),
)


tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(imaging=masked_imaging, tracer=tracer)
fit.log_evidence

"""
__Model__

We compose our lens model using `GalaxyModel` objects, which represent the galaxies we fit to our data. In this 
example our lens model is:

 - An `EllIsothermal` `MassProfile` for the lens `Galaxy`'s mass (5 parameters).
 - A `Rectangular` `Pixelization`.which reconstructs the source `Galaxy`'s light. We will fix its resolution to 
   30 x 30 pixels, which balances fast-run time with sufficient resolution to reconstruct its light. (0 parameters).
 - A `Constant` `Regularization`.scheme which imposes a smooothness prior on the source reconstruction (1 parameter). 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=1. 

It is worth noting the `Pixelization` and `Regularization` use significantly fewer parameter (1 parameters) than 
fitting the source using `LightProfile`'s (7+ parameters). 

NOTE: By default, **PyAutoLens** assumes the image has been reduced such that the lens galaxy centre is at (0.0", 0.0"),
with the priors on the lens `MassProfile` coordinates set accordingly. if for your dataset the lens is not centred at 
(0.0", 0.0"), we recommend you reduce your data so it is (see `autolens_workspace/preprocess`). Alternatively, you can 
manually override the priors (see `autolens_workspace/examples/customize/priors.py`).
"""
bulge = al.lp.EllSersic(
    centre=(0.0, 0.0),
    elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=4.0,
    effective_radius=0.6,
    sersic_index=3.0,
)
disk = al.lp.EllExponential(
    centre=(0.0, 0.0),
    elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, angle=30.0),
    intensity=2.0,
    effective_radius=1.6,
)

mass = af.Model(al.mp.EllIsothermal)

mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.02, upper_limit=0.02)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.02, upper_limit=0.02)
mass.elliptical_comps.elliptical_comps_0 = af.UniformPrior(
    lower_limit=0.05, upper_limit=0.15
)
mass.elliptical_comps.elliptical_comps_1 = af.UniformPrior(
    lower_limit=-0.05, upper_limit=0.05
)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, disk=disk, mass=mass)
source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=pixelization_shape_2d),
    regularization=al.reg.Constant,
)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search__

The lens model is fitted to the data using a `NonLinearSearch`. In this example, we use the
nested sampling algorithm Dynesty (https://dynesty.readthedocs.io/en/latest/).

The script `autolens_workspace/examples/model/customize/non_linear_searches.py` gives a description of the types of
non-linear searches that **PyAutoLens** supports. If you do not know what a `NonLinearSearch` is or how it 
operates, checkout chapters 1 and 2 of the HowToLens lecture series.

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/examples/beginner/mass_sie__source_sersic/phase_mass[sie]_source[inversion]`.
"""
number_of_cores = int(sys.argv[1])

search = af.DynestyStatic(
    path_prefix=path.join("profiling", "inversion_magnification", instrument),
    name=f"x{number_of_cores}_mass[sie]_source[inversion]",
    maxcall=200,  # This sets how long the model-fit takes.
    nlive=50,
    number_of_cores=number_of_cores,
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

import cProfile

cProfile.run(
    "search.fit(model=model, analysis=analysis)",
    f"x{int(number_of_cores)}_{instrument}_model_inversion_magnification.prof",
)

# result = search.fit(model=model, analysis=analysis)

import sys

sys.exit()
