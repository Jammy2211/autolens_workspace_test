"""
Overview: Fit
-------------

**PyAutoLens** uses `Tracer` objects to represent a strong lensing system. Now, we`re going use these objects to
fit `Imaging` data of a strong lens.

The `autolens_workspace` comes distributed with simulated images of strong lenses (an example of how these simulations
are made can be found in the `simulate.py` example, with all simulator scripts located in `autolens_workspac/simulators`.
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
__Loading Data__

We we begin by loading the strong lens dataset `mass_sie__source_sersic` from .fits files:

Load the strong lens dataset `light_sersic__mass_sie__source_sersic` from .fits files, which is the dataset 
we will use to demosntrate fitting.
"""
dataset_name = "mass_sie__source_sersic_compact"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.05,
)

"""
The `Imaging` mat_plot_2d also contains a subplot which plots all these properties simultaneously.
"""
imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Masking__

We now need to mask the data, so that regions where there is no signal (e.g. the edges) are omitted from the fit.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

"""
__Fitting With Sub 1__
"""
imaging = imaging.apply_mask(mask=mask)

imaging = imaging.apply_settings(
    settings=al.SettingsImaging(grid_class=al.Grid2D, sub_size=1)
)

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
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=0.1,
        sersic_index=3.0,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[source_galaxy, lens_galaxy])

fit = al.FitImaging(dataset=imaging, tracer=tracer)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)
fit_imaging_plotter.subplot_fit_imaging()

print(fit.figure_of_merit)


"""
__Fitting With Grid Iterate__
"""
imaging = imaging.apply_settings(
    settings=al.SettingsImaging(grid_class=al.Grid2DIterate)
)

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
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=0.1,
        sersic_index=3.0,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy])

fit = al.FitImaging(dataset=imaging, tracer=tracer)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)
fit_imaging_plotter.subplot_fit_imaging()

print(fit.figure_of_merit)
