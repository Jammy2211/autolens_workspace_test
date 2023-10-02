"""
Overview: Multi-Wavelength
--------------------------

**PyAutoLens** supports the analysis of multiple datasets simultaneously, including many CCD imaging datasets
observed at different wavebands (e.g. red, blue, green) and combining imaging and interferometer datasets.

This enables multi-wavelength lens modeling, whereby the color of the lens and source galaxies varies across
the datasets.

It also allows one to fit images of the same lens at the same wavelength simultaneously, for example if one opted
to analysis images of a strong lens before they are combined to a single frame via the multidrizzling data reduction
process.

Multi-wavelength lens modeling offers a number of advantages:

- It provides a wealth of additional information to fit the lens model, especially if the source changes its
appears across wavelength.

- It overcomes challenges associated with the lens and source galaxy emission blending with one another, as their
 brightness depends differently on wavelength.

- Instrument systematic effects, for example an uncertain PSF, will impact the model less because they vary across
 each dataset.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")
from os import path
import os

workspace_path = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(workspace_path, "config", "searches"))

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Colors__

For multi-wavelength imaging datasets, we begin by defining the colors of the multi-wavelength images. 

For this overview we use only two colors, green (g-band) and red (r-band), but extending this to more datasets
is straight forward.
"""
color_list = ["g", "r"]

"""
__Pixel Scales__

Every dataset in our multi-wavelength observations can have its own unique pixel-scale.
"""
pixel_scales_list = [0.08, 0.12]

"""
__Dataset__

Multi-wavelength imaging datasets do not use any new objects or class in PyAutoLens.

We simply use lists of the classes we are now familiar with, for example the `Imaging` class.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "with_lens_light"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

dataset_list = [
    al.Imaging.from_fits(
        data_path=path.join(dataset_path, f"{color}_image.fits"),
        psf_path=path.join(dataset_path, f"{color}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{color}_noise_map.fits"),
        pixel_scales=pixel_scales,
    )
    for color, pixel_scales in zip(color_list, pixel_scales_list)
]

"""
Here is what our r-band and g-band observations of this lens system looks like.

Now how in the r-band, the lens outshines the source, whereas in the g-band the source galaxy is more visible. 

The different variation of the colors of the lens and source across wavelength is a powerful tool for lensing modeling,
as it helps PyAutoLens deblend the two objects.
"""
for imaging, color in zip(dataset_list, color_list):
    mat_plot_2d = aplt.MatPlot2D(
        title=aplt.Title(label=f"{color}-band Image"),
        output=aplt.Output(
            path=workspace_path, filename=f"{color}_image", format="png"
        ),
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot_2d)
    dataset_plotter.figures_2d(data=True)

"""
__Mask__

The model-fit requires a `Mask2D` defining the regions of the image we fit the lens model to the data, which we define
and use to set up the `Imaging` object that the lens model fits.

For multi-wavelength lens modeling, we use the same mask for every dataset whenever possible. This is not absolutely 
necessary, but provides a more reliable analysis.
"""
mask_list = [
    al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )
    for dataset in dataset_list
]


dataset_list = [
    dataset.apply_mask(mask=mask) for imaging, mask in zip(dataset_list, mask_list)
]

for dataset in dataset_list:
    mat_plot_2d = aplt.MatPlot2D(
        title=aplt.Title(label=f"{color}-band Image"),
        output=aplt.Output(
            path=workspace_path, filename=f"{color}_masked_image", format="png"
        ),
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot_2d)
    dataset_plotter.figures_2d(data=True)

"""
__Analysis__

We create a list of `AnalysisImaging` objects for every dataset.
"""
analysis_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

"""
We now introduce the key new aspect to the PyAutoLens API in this overview, which is critical to fitting
multiple datasets simultaneously.

We sum this list of analysis objects to create an overall `CombinedAnalysis` object, which we can use to fit the 
multi-wavelength imaging data, where:

 - The log likelihood function of this summed analysis class is the sum of the log likelihood functions of each 
 individual analysis objects (e.g. the fit to each separate waveband).

 - The summing process ensures that tasks such as outputting results to hard-disk, visualization, etc use a 
 structure that separates each analysis and therefore each dataset.
"""
analysis = sum(analysis_list)

"""
We can parallelize the likelihood function of these analysis classes, whereby each evaluation is performed on a 
different CPU.
"""
analysis.n_cores = 1

"""
__Model__

We compose the lens model as per usual.
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=al.lp.Sersic,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
However, there is a problem for multi-wavelength datasets. Should the light profiles of the lens's bulge and
source's bulge have the same parameters for each wavelength image?

The answer is no. At different wavelengths, different stars appear brighter or fainter, meaning that the overall
appearance of the lens and source galaxies will change. 

We can therefore allow specific light profile parameters to vary across wavelength, and therefore act as free
parameters in the fit to each image. 

We do this using the combined analysis object as follows:
"""
analysis = analysis.with_free_parameters(
    model.galaxies.lens.bulge.intensity, model.galaxies.source.bulge.intensity
)

"""
In this simple overview, this has added two additional free parameters to the model whereby:

 - The lens bulge's intensity is different in both multi-wavelength images.
 - The source bulge's intensity is different in both multi-wavelength images.
 
It is entirely plausible that more parameters should be free to vary across wavelength. 

This choice ultimately depends on the quality of data being fitted and intended science goal. Regardless, it is clear
how the above API can be extended to add any number of additional free parameters.

__Search + Model Fit__

Fitting the model uses the same API we introduced in previous overviews.
"""
search = af.DynestyStatic(
    name="overview_example_multiwavelength_3", nlive=200, walks=10
)

"""
The result object returned by this model-fit is a list of `Result` objects, because we used a combined analysis.
Each result corresponds to each analysis, and therefore corresponds to the model-fit at that wavelength.
"""
result_list = search.fit(model=model, analysis=analysis)

"""
Plotting each result's tracer shows that the lens and source galaxies appear different in each result, owning to their 
different intensities.
"""

# print(result_list[0].max_log_likelihood_instance.galaxies.source.bulge.intensity)
# print(result_list[1].max_log_likelihood_instance.galaxies.source.bulge.intensity)
#
# print(result_list[0].max_log_likelihood_tracer.galaxies[1].bulge.intensity)
# print(result_list[1].max_log_likelihood_tracer.galaxies[1].bulge.intensity)
#
# stop
for result, color in zip(result_list, color_list):
    mat_plot_2d = aplt.MatPlot2D(
        title=aplt.Title(label=f"Lens and source {color}-band Images"),
        output=aplt.Output(
            path=workspace_path, filename=f"{color}_tracer_image", format="png"
        ),
    )

    include_2d = aplt.Include2D(
        border=False,
        light_profile_centres=False,
        mass_profile_centres=False,
        tangential_critical_curves=False,
        radial_critical_curves=False,
        tangential_caustics=False,
        radial_caustics=False,
    )

    tracer_plotter = aplt.TracerPlotter(
        tracer=result.max_log_likelihood_tracer,
        grid=result.grid,
        mat_plot_2d=mat_plot_2d,
        include_2d=include_2d,
    )
    tracer_plotter.subplot_lensed_images()


"""
__Wavelength Dependence__

In the example above, a free `intensity` parameter is created for every multi-wavelength dataset. This would add 5+ 
free parameters to the model if we had 5+ datasets, quickly making a complex and difficult to fit model parameterization.

We can instead parameterize the intensity of the lens and source galaxies as a user defined function of 
wavelength, for example following a relation `y = (m * x) + c` -> `intensity = (m * wavelength) + c`.

By using a linear relation `y = mx + c` the free parameters are `m` and `c`, which does not scale with the number
of datasets. For datasets with multi-wavelength images (e.g. 5 or more) this allows us to parameterize the variation 
of parameters across the datasets in a way that does not lead to a very complex parameter space.

Below, we show how one would do this for the `intensity` of a lens galaxy's bulge, give three wavelengths corresponding
to a dataset observed in the g and I bands.
"""
wavelength_list = [464, 658, 806]

m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)
linear = af.Add(af.Multiply(wavelength_list, m), c)

analysis_list = [
    al.AnalysisImaging(dataset=dataset).with_model(linear) for dataset in dataset_list
]

"""
__Same Wavelength Datasets__

The above API can fit multiple datasets which are observed at the same wavelength.

An example use case might be analysing undithered images (e.g. from HST) before they are combined via the 
multidrizzing process, to remove correlated noise in the data.

The pointing of each observation, and therefore centering of each dataset, may vary in an unknown way. This
can be folded into the model and fitted for as follows:

TODO : ADD CODE EXAMPLE.

__Interferometry and Imaging__

The above API can combine modeling of imaging and interferometer datasets (see ? for an example script
showing this in full).

Below are mock strong lens images of a system observed at a green wavelength (g-band) and with an interferometer at
sub millimeter wavelengths. 

A number of benefits are apparently if we combine the analysis of both datasets at both wavelengths:

 - The lens galaxy is invisible at sub-mm wavelengths, making it straight-forward to infer a lens mass model by
 fitting the source at submm wavelengths.
 
 - The source galaxy appears completely different in the g-band and at sub-millimeter wavelengths, providing a lot
 more information with which to constrain the lens galaxy mass model.
"""

"""
__Interferometer Masking__

We define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=4.0, sub_size=1
)

"""
__Interferometer Dataset__

Load and plot the strong lens `Interferometer` dataset `mass_sie__source_sersic` from .fits files, which we will fit 
with the lens model.
"""
dataset_type = "multi"
dataset_label = "interferometer"
dataset_name = "no_lens_light"
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

mat_plot_2d = aplt.MatPlot2D(
    title=aplt.Title(label=f"Interferometer Dirty Image"),
    output=aplt.Output(path=workspace_path, filename=f"dirty_image", format="png"),
)

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset, mat_plot_2d=mat_plot_2d)
dataset_plotter.figures_2d(dirty_image=True)

"""
__Imaging Dataset__
"""
# dataset_type = "multi"
# dataset_label = "imaging"
# dataset_name = "with_lens_light"
# dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)
#
# dataset = al.Imaging.from_fits(
#     data_path=path.join(dataset_path, "g_image.fits"),
#     psf_path=path.join(dataset_path, "g_psf.fits"),
#     noise_map_path=path.join(dataset_path, "g_noise_map.fits"),
#     pixel_scales=0.08,
# )
#
# mask = al.Mask2D.circular(
#     shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
# )
#
# dataset = dataset.apply_mask(mask=mask)
#
# mat_plot_2d = aplt.MatPlot2D(
#     title=aplt.Title(label=f"{color}-band Image"),
#     output=aplt.Output(path=workspace_path, filename=f"{color}_image", format="png")
# )
#
# dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
# dataset_plotter.subplot_dataset()
