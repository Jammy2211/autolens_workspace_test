from os import path
import autolens as al
import autolens.plot as aplt

"""
This example illustrates how to plot an `FitImaging` object using an `FitImagingPlotter`.

First, lets load example imaging of of a strong lens as an `Imaging` object.
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
We now mask the data and fit it with a `Tracer` to create a `FitImaging` object.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    sub_size=1,
    radius=3.0,
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalPowerLaw(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=45.0),
        slope=2.0,
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.0, 0.05)),
)

"""
We can also plot a `FitImaging` which uses an `Inversion`.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(25, 25)),
    regularization=al.reg.Constant(coefficient=1.0),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

"""
The `plane_image_from_plane` method now plots the the reconstructed source on the Voronoi pixel-grid. It can use the
`Include2D` object to plot the `Mapper`'s specific structures like the image and source plane pixelization grids.
"""
include_2d = aplt.Include2D(
    mapper_data_pixelization_grid=True, mapper_source_grid_slim=True
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_plotter.figures_of_planes(plane_image=True, plane_index=1)
