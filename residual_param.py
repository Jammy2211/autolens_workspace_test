"""
__PROFILING: Inversion Voronoi__

This profiling script times how long it takes to fit `Imaging` data with a `Voronoi` pixelization for
datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoLens** log likelihood function.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import time
import json
import autolens as al
import autolens.plot as aplt


"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 1
mask_radius = 4.0
psf_shape_2d = (21, 21)
pixels = 1000

use_positive_only_solver = True
maxiter = 5000


"""
The lens galaxy used to fit the data, which is identical to the lens galaxy used to simulate the data. 
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=4.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=2.0,
        effective_radius=1.6,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.001, gamma_2=0.001),
)


instrument = "hst"

pixel_scale = 0.05

"""
Load the dataset for this instrument / resolution.
"""
dataset_path = path.join("dataset", "imaging", "instruments", instrument)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scale,
    over_sample_size_pixelization=1,
)

"""
Apply the 2D mask, which for the settings above is representative of the masks we typically use to model strong lenses.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

masked_dataset = dataset.apply_mask(mask=mask)

"""
Generate the adapt-images used to adapt the source pixelization and regularization.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)
lens_adapt_data = lens_galaxy.image_2d_from(grid=masked_dataset.grid)
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
traced_grid = tracer.traced_grid_2d_list_from(grid=masked_dataset.grid)[1]
source_adapt_data = source_galaxy.image_2d_from(grid=traced_grid)

"""
The source galaxy whose `VoronoiBrightness` `Pixelization` fits the data.
"""
pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Hilbert(pixels=pixels, weight_floor=0.2, weight_power=3.0),
    mesh=al.mesh.Voronoi(),
    regularization=al.reg.AdaptiveBrightnessSplit(
        inner_coefficient=0.01, outer_coefficient=100.0, signal_scale=0.05
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

adapt_images = al.AdaptImages(
    galaxy_image_dict={
        lens_galaxy: lens_adapt_data,
        source_galaxy: source_adapt_data,
    }
)


"""
__Numba Caching__

Call FitImaging once to get all numba functions initialized.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(
    dataset=masked_dataset,
    tracer=tracer,
    adapt_images=adapt_images,
)


file_path = "."

"""
Output an image of the fit, so that we can inspect that it fits the data as expected.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.001, gamma_2=0.001),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

grid = al.Grid2D.from_mask(mask=mask, over_sample_size=1)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

image_prev = tracer.image_2d_from(
    grid=grid,
)


for sub_size in range(8):
    grid = al.Grid2D.from_mask(
        mask=mask, over_sampling=al.OverSampling(sub_size=sub_size)
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    image_new = tracer.image_2d_from(
        grid=grid,
    )

    mat_plot_2d = aplt.MatPlot2D(
        output=aplt.Output(
            path=file_path, filename=f"image_sub_{sub_size}", format="png"
        )
    )
    plotter = aplt.Array2DPlotter(array=image_new, mat_plot_2d=mat_plot_2d)
    plotter.figure_2d()

    residuals = image_new - image_prev

    mat_plot_2d = aplt.MatPlot2D(
        output=aplt.Output(
            path=file_path, filename=f"residuals_{sub_size}", format="png"
        )
    )
    plotter = aplt.Array2DPlotter(array=residuals, mat_plot_2d=mat_plot_2d)
    plotter.figure_2d()

    image_prev = image_new
