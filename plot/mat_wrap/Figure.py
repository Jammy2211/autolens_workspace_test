from os import path
import autolens as al
import autolens.plot as aplt

"""
This example illustrates how to customize the Matplotlib figure window that displays PyAutoLens figures and 
subplots.

First, lets load an example Hubble Space Telescope image of a real strong lens as an `Array2D`.
"""

dataset_path = path.join("dataset", "slacs", "slacs1430+4105")
image_path = path.join(dataset_path, "image.fits")
image = al.Array2D.from_fits(file_path=image_path, hdu=0, pixel_scales=0.03)

"""
We can customize the figure using the `Figure` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.figure.html
"""

figure = aplt.Figure(
    figsize=(7, 7),
    dpi=100.0,
    facecolor="white",
    edgecolor="black",
    frameon=True,
    clear=False,
    tight_layout=False,
    constrained_layout=False,
)

mat_plot_2d = aplt.MatPlot2D(figure=figure)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure()

"""
We can also customize the aspect ratio of the image displayed in a figure by passing the `Figure` an aspect ratio. 

This customizes the aspect ratio when the method `plt.imshow` is called.

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html
"""

figure = aplt.Figure(aspect="square")

mat_plot_2d = aplt.MatPlot2D(figure=figure)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure()
