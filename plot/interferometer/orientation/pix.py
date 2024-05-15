"""
Plots: FitInterferometerPlotter
===============================

This example illustrates how to plot an `FitInterferometer` object using an `FitInterferometerPlotter`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

First, lets load example interferometer of of a strong lens as an `Interferometer` object.
"""
dataset_type = "interferometer"
dataset_name = "orientation"
dataset_path = path.join("dataset", dataset_type, dataset_name)

real_space_mask = al.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=4.0, centre=(0.5, 0.5)
)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

"""
__Fit__

We now mask the data and fit it with a `Tracer` to create a `FitInterferometer` object.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.5, 0.5),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.Pixelization(
        image_mesh=al.image_mesh.Overlay(shape=(40, 40)),
        regularization=al.reg.Constant(coefficient=10.0),
        mesh=al.mesh.Delaunay(),
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitInterferometer(dataset=dataset, tracer=tracer)

"""
__Output__
"""
output = aplt.Output(
    path=path.join("plot", "interferometer", "orientation", "plots", "pix_delaunay"),
    format="png",
)

"""
__Figures__

We now pass the FitInterferometer to an `FitInterferometerPlotter` and call various `figure_*` methods 
to plot different attributes.
"""
fit_plotter = aplt.FitInterferometerPlotter(
    fit=fit,
    mat_plot_1d=aplt.MatPlot1D(output=output),
    mat_plot_2d=aplt.MatPlot2D(output=output),
)

"""
__Subplots__

The `FitInterferometerPlotter` may also plot a subplot of these attributes.
"""
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()


"""
The plane images can be combined to plot the appearance of the galaxy in real-space.
"""
fit_plotter.subplot_fit_real_space()
