"""
Simulator: Double Einstein Ring
===============================

This script simulates `Imaging` of a 'galaxy-scale' strong lens where there are two source's at different different
redshifts, creating a double einstein ring. Specifically:

 - The first lens galaxy's light is an `EllSersic` and total mass distribution is an `EllIsothermal`.
 - The second galaxy, which is the first source galaxy,  total mass distribution is an `EllIsothermal` and its light is
 an `EllSersic`.
 - The third galaxy, which is the second source galaxy, light profile is an `EllSersic`.
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
__Dataset Paths__

The `dataset_type` describes the type of data being simulated (in this case, `Imaging` data) and `dataset_name`
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/image.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/psf.fits`.
"""
dataset_type = "imaging"
dataset_label = "with_lens_light"
dataset_name = "light_sersic_mass_sis__source_sis_exp__source_exp"

"""
The path where the dataset will be output, which in this case is:
`/autolens_workspace/dataset/imaging/with_lens_light/light_sersic_mass_sis__source_sis_exp__source_exp`
"""
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

"""
__Simulate__

For simulating an image of a strong lens, we recommend using a Grid2DIterate object. This represents a grid of (y,x) 
coordinates like an ordinary Grid2D, but when the light-profile`s image is evaluated below (using the Tracer) the 
sub-size of the grid is iteratively increased (in steps of 2, 4, 8, 16, 24) until the input fractional accuracy of 
99.99% is met.

This ensures that the divergent and bright central regions of the source galaxy are fully resolved when determining the
total flux emitted within a pixel.
"""
grid = al.Grid2DIterate.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
    fractional_accuracy=0.9999,
    sub_steps=[2, 4, 8, 16, 24],
)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

"""
__Ray Tracing__

Setup the lens galaxy's mass (SIE+Shear) and source galaxy light (elliptical Sersic) for this simulated lens.

For lens modeling, defining ellipticity in terms of the `elliptical_comps` improves the model-fitting procedure.

However, for simulating a strong lens you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle, where angle is
in degrees and defined counter clockwise from the positive x-axis.

We can use the **PyAutoLens** `convert` module to determine the elliptical components from the axis-ratio and angle.
"""

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.5,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SphExponential(
        centre=(-0.15, -0.15), intensity=1.2, effective_radius=0.1
    ),
    mass=al.mp.SphIsothermal(centre=(-0.15, -0.15), einstein_radius=0.6),
)

source_galaxy_1 = al.Galaxy(
    redshift=2.0,
    bulge=al.lp.SphExponential(
        centre=(-0.45, 0.35), intensity=0.6, effective_radius=0.07
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1]
)

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
imaging.output_to_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.
"""
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

imaging_plotter = aplt.ImagingPlotter(imaging=imaging, mat_plot_2d=mat_plot_2d)
imaging_plotter.subplot_imaging()
imaging_plotter.figures_2d(image=True)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d)
tracer_plotter.subplot_tracer()

"""
For a double Einstein ring we output the image of the two source galaxies individually,
as this can be difficult to discern from the final image.
"""
mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(path=dataset_path, filename="source_0_image", format="png")
)
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d)
plane_plotter = tracer_plotter.plane_plotter_from(plane_index=1)
plane_plotter.figures_2d(image=True)
mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(path=dataset_path, filename="source_1_image", format="png")
)
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d)
plane_plotter = tracer_plotter.plane_plotter_from(plane_index=2)
plane_plotter.figures_2d(image=True)

"""
__Tracer Output__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `Tracer.from_json`.
"""
tracer.output_to_json(file_path=path.join(dataset_path, "tracer.json"))

"""
The dataset can be viewed in the 
folder `autolens_workspace/imaging/with_lens_light/light_sersic_mass_sis__source_sis_exp__source_exp`.
"""
