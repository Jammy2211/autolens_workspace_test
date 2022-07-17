"""
Plots: InversionPlotter
=======================

This example illustrates how to plot a `Inversion` using a `InversionPlotter`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

First, lets load example imaging of of a strong lens as an `Imaging` object.
"""
dataset_name = "mass_sie__source_sersic__offset"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
We now mask the `Imaging` data so we can fit it with an `Inversion`.
"""
mask_2d = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=4.0
)
imaging = imaging.apply_mask(mask=mask_2d)

"""
__Tracer__

The `Inversion` maps pixels from the image-plane of our `Imaging` data to its source plane, via a lens model.

Lets create a `Tracer` which we will use to create the `Inversion`.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.05, 0.05)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(25, 25)),
    regularization=al.reg.Constant(coefficient=1.0),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

"""
We can extract the `Inversion` from the `Tracer` by passing it the imaging data.
"""
inversion = tracer.to_inversion.inversion_imaging_from(
    dataset=imaging,
    image=imaging.image,
    noise_map=imaging.noise_map,
    w_tilde=imaging.w_tilde,
)

"""
__Figures__

We now pass the inversion to a `InversionPlotter` and call various `figure_*` methods to plot different attributes.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.figures_2d(reconstructed_image=True)

"""
An `Inversion` can have multiple mappers, which reconstruct multiple source galaxies at different redshifts and
planes (e.g. double Einstein ring systems).

To plot an individual source we must therefore specify the mapper index of the source we plot.
"""
mat_plot_2d = aplt.MatPlot2D(
    #   output=aplt.Output(path=".", filename="inversion_source", format="png")
)

inversion_plotter = aplt.InversionPlotter(inversion=inversion, mat_plot_2d=mat_plot_2d)

inversion_plotter.figures_2d_of_mapper(mapper_index=0, reconstruction=True)

"""
Finish.
"""
