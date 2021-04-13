"""
__Example: Modeling__

Model-fitting is handled by our project **PyAutoFit**, a probabilistic programming language for non-linear model
fitting. The setting up of configuration files is performed by our project **PyAutoConf**. we'll need to import
both to perform the model-fit.
"""

"""
In this script, we fit `Imaging` of a strong lens system where:

 - The lens `Galaxy`'s light is omitted (and is not present in the simulated data).
 - The lens `Galaxy`'s total mass distribution is an `EllIsothermal`.
 - The source `Galaxy`'s surface-brightness is modeled using an `Inversion`.

An `Inversion` reconstructs the source`s light using a pixel-grid, which is regularized using a prior that forces
this reconstruction to be smooth. This uses `Pixelization` and `Regularization` objects and in this example we will
use their simplest forms, a `Rectangular` `Pixelization` and `Constant` `Regularization`.scheme.

Inversions are covered in detail in chapter 4 of the **HowToLens** lectures.
"""

"""
Load the strong lens dataset `mass_sie__source_sersic` `from .fits files, , which we fit with the lens model.

This is the same dataset we fitted in the `autolens/intro/fitting.py` example.
"""

"""
Manually turn off test-mode so lens model is fitted fully.
"""
from autoconf import conf

conf.instance["general"]["test"]["test_mode"] = False

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

# instrument = "vro"
# instrument = "euclid"
instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

dataset_path = path.join("dataset", "imaging", instrument)

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

imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
__numba initialization__
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
__Phase__

To perform lens modeling, we create a `PhaseImaging` object, which comprises:

   - The `GalaxyModel`'s used to fit the data.
   - The `SettingsPhase` which customize how the model is fitted to the data.
   - The `NonLinearSearch` used to sample parameter space.

Once we have create the phase, we `run` it by passing it the data and mask.

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
search = af.DynestyStatic(
    path_prefix=path.join("profiling", "inversion_magnification", instrument),
    name="phase_mass[sie]_source[inversion]",
    maxcall=1000,  # This sets how long the model-fit takes.
    nlive=50,
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

### BEGIN PROFILING HERE ###

result = search.fit(model=model, analysis=analysis)

"""
The phase above returned a result, which, for example, includes the lens model corresponding to the maximum
log likelihood solution in parameter space.
"""
print(result.max_log_likelihood_instance)

"""
It also contains instances of the maximum log likelihood Tracer and FitImaging, which can be used to visualize
the fit.
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=mask.masked_grid_sub_1
)
tracer_plotter.subplot_tracer()
fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()

"""
Checkout `/path/to/autolens_workspace/examples/model/results.py` for a full description of the result object.
"""
