"""
Overview: Interferometer
------------------------

Alongside CCD `Imaging` data, **PyAutoLens** supports the modeling of interferometer data from submillimeter and radio
observatories.

The dataset is fitted directly in the uv-plane, circumventing issues that arise when fitting a `dirty image` such as
correlated noise.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autolens as al
import autolens.plot as aplt
from os import path
import numpy as np
import os

workspace_path = os.getcwd()

"""
__Real Space Mask__

To begin, we define a real-space mask. Although interferometer lens modeling is performed in the uv-plane and 
therefore Fourier space, we still need to define the grid of coordinates in real-space from which the lensed source's 
images are computed. It is this image that is mapped to Fourier space to compare to the uv-plane data.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(400, 400), pixel_scales=0.025, radius=3.0
)

"""
__Dataset__

We next load an interferometer dataset from fits files, which follows the same API that we have seen for an `Imaging`
object.
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "interferometer", dataset_name)

interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

"""
The PyAutoLens plot module has tools for plotting interferometer datasets, including the visibilities, noise-map
and uv wavelength which represent the interferometer`s baselines. 

The data used in this tutorial contains only ~300 visibilities and is representative of a low resolution 
Square-Mile Array (SMA) dataset. 

We made this choice so the script runs fast, and we discuss below how PyAutoLens can scale up to large visibilities
datasets from an instrument like ALMA.
"""
interferometer_plotter = aplt.InterferometerPlotter(
    interferometer=interferometer,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="visibilities", format="png")
    ),
)
interferometer_plotter.figures_2d(visibilities=True)

interferometer_plotter = aplt.InterferometerPlotter(
    interferometer=interferometer,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="uv_wavelengths", format="png")
    ),
)
interferometer_plotter.figures_2d(uv_wavelengths=True)

interferometer_plotter = aplt.InterferometerPlotter(
    interferometer=interferometer,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="dirty_image", format="png")
    ),
)

interferometer_plotter.figures_2d(dirty_image=True)

interferometer_plotter = aplt.InterferometerPlotter(
    interferometer=interferometer,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(
            path=workspace_path, filename="dirty_signal_to_noise", format="png"
        )
    ),
)

interferometer_plotter.figures_2d(dirty_signal_to_noise_map=True)


"""
__Tracer__

To perform uv-plane modeling, **PyAutoLens** generates an image of the strong lens system in real-space via a `Tracer`. 

Lets quickly set up the `Tracer` we'll use in this example.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma=(0.05, 0.05)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer,
    grid=real_space_mask.masked_grid_sub_1,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="image_pre_ft", format="png")
    ),
)
tracer_plotter.figures_2d(image=True)

"""
__UV-Plane__

To perform uv-plane modeling, **PyAutoLens** next Fourier transforms this image from real-sapce to the uv-plane.
This operation uses a *Transformer* object, of which there are multiple available in **PyAutoLens**. This includes
a direct Fourier transform which performs the exact Fourier transformw without approximation.
"""
transformer_class = al.TransformerDFT

"""
However, the direct Fourier transform is inefficient. For ~10 million visibilities, it requires thousands of seconds
to perform a single transform. This approach is therefore unfeasible for high quality ALMA and radio datasets.

For this reason, **PyAutoLens** supports the non-uniform fast fourier transform algorithm
**PyNUFFT** (https://github.com/jyhmiinlin/pynufft), which is significantly faster, being able too perform a Fourier
transform of ~10 million in less than a second!
"""
transformer_class = al.TransformerNUFFT

"""
The use this transformer in a fit, we use the `apply_settings` method.
"""
interferometer = interferometer.apply_settings(
    settings=al.SettingsInterferometer(transformer_class=transformer_class)
)

"""
__Fitting__

The interferometer can now be used with a `FitInterferometer` object to fit it to a data-set:
"""
fit = al.FitInterferometer(dataset=interferometer, tracer=tracer)

fit_interferometer_plotter = aplt.FitInterferometerPlotter(fit=fit)

"""
Visualization of the fit can be performed in the uv-plane or in real-space. 

Note that the fit is not performed in real-space, but plotting it in real-space is often more informative.
"""
fit_interferometer_plotter.subplot_fit_interferometer()
fit_interferometer_plotter.subplot_fit_real_space()

"""
Interferometer data can also be modeled using pixelized source's, which again perform the source reconstruction by
directly fitting the visibilities in the uv-plane.

The source reconstruction can be visualized in real space:
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.mesh.DelaunayMagnification(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=1.0),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitInterferometer(
    dataset=interferometer,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(use_linear_operators=True),
)

fit_interferometer_plotter = aplt.FitInterferometerPlotter(
    fit=fit,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(
            path=workspace_path, filename="model_visibilities", format="png"
        )
    ),
)
fit_interferometer_plotter.figures_2d(model_visibilities=True)
fit_interferometer_plotter = aplt.FitInterferometerPlotter(
    fit=fit,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(
            path=workspace_path, filename="fit_dirty_images", format="png"
        )
    ),
)
fit_interferometer_plotter.subplot_fit_dirty_images()

"""
__Efficiency__

Computing this source recontruction would be extremely inefficient if **PyAutoLens** used a traditional approach to
linear algebra which explicitly stored in memory the values required to solve for the source fluxes. In fact, for an
interferomter dataset of ~10 million visibilities this would require **hundreds of GB of memory**!

**PyAutoLens** uses the library **PyLops** (https://pylops.readthedocs.io/en/latest/) to represent this calculation as
a sequence of memory-light linear operators.
"""
inversion_plotter = aplt.InversionPlotter(inversion=fit.inversion)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

"""
__Lens Modeling__

It is straight forward to fit a lens model to an interferometer dataset, using the same API that we saw for imaging
data in the `overview/modeling.py` example.

__Model__

We first compose the model, in the same way described in the `modeling.py` overview script:
"""
lens_galaxy_model = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)

source_galaxy_model = af.Model(al.Galaxy, redshift=1.0, disk=al.lp.Exponential)

galaxies = af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)
model = af.Collection(galaxies=galaxies)

"""
__Non-linear Search__

We again choose the non-linear search `dynesty` (https://github.com/joshspeagle/dynesty).
"""
search = af.DynestyStatic(name="overview_interferometer")

"""
__Analysis__

Whereas we previously used an `AnalysisImaging` object, we instead use an `AnalysisInterferometer` object which fits 
the lens model in the correct way for an interferometer dataset. 

This includes mapping the lens model from real-space to the uv-plane via the Fourier transform discussed above.
"""
analysis = al.AnalysisInterferometer(dataset=interferometer)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

The results can be found in the `output/overview_interferometer` folder in the `autolens_workspace`.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The **PyAutoLens** visualization library and `FitInterferometer` object includes specific methods for plotting the 
results.
"""
fit_interferometer_plotter = aplt.FitInterferometerPlotter(
    fit=result.max_log_likelihood_fit
)
fit_interferometer_plotter.subplot_fit_interferometer()
fit_interferometer_plotter.subplot_fit_dirty_images()

"""
__Simulation__3

Simulated interferometer datasets can be generated using the ``SimulatorInterferometer`` object, which includes adding
Gaussian noise to the visibilities:
"""
simulator = al.SimulatorInterferometer(
    uv_wavelengths=interferometer.uv_wavelengths, exposure_time=300.0, noise_sigma=0.01
)

real_space_grid_2d = al.Grid2D.uniform(
    shape_native=real_space_mask.shape_native, pixel_scales=real_space_mask.pixel_scales
)

interferometer = simulator.via_tracer_from(tracer=tracer, grid=real_space_grid_2d)

"""
__Wrap Up__

The `interferometer` package of the `autolens_workspace` contains numerous example scripts for performing 
interferometer modeling and simulating strong lens interferometer datasets.
"""
