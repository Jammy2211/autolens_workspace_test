from os import path
import autolens as al
import autolens.plot as aplt

"""
This example illustrates how to customize the ticks on a Colorbar in PyAutoLens figures and subplots.

First, lets load an example Hubble Space Telescope image of a real strong lens as an `Array2D`.
"""

dataset_path = path.join("dataset", "slacs", "slacs1430+4105")
image_path = path.join(dataset_path, "image.fits")
image = al.Array2D.from_fits(file_path=image_path, hdu=0, pixel_scales=0.03)

"""
We can customize the colorbar ticks using the `ColorbarTickParams` matplotlib wrapper object which wraps the 
following method of the matplotlib colorbar:

 https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html
"""

colorbar_tickparams = aplt.ColorbarTickParams(
    axis="both",
    reset=False,
    which="major",
    direction="in",
    length=2,
    width=2,
    color="r",
    pad=0.1,
    labelsize=10,
    labelcolor="r",
)

mat_plot_2d = aplt.MatPlot2D(colorbar_tickparams=colorbar_tickparams)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure()
