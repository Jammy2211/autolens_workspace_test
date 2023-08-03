import autolens as al
import matplotlib.pyplot as plt

grid_2d = al.Grid2DIterate.uniform(
    shape_native=(160, 160), pixel_scales=0.05, fractional_accuracy=0.9999
)
psf = al.Kernel2D.from_gaussian(
    shape_native=(21, 21), sigma=0.05, pixel_scales=grid_2d.pixel_scales, normalize=True
)  # Define grid, psf, and simulator for Euclid as taken from the autolens workspace
simulator = al.SimulatorImaging(
    exposure_time=9999 * 2260.0,
    psf=psf,
    background_sky_level=1.0,
    add_poisson_noise=True,
)


source_dict = {}

source_dict["bulge"] = al.lp.Sersic(
    centre=(0, 0),
    ell_comps=al.convert.ell_comps_from(
        axis_ratio=1, angle=0
    ),  # Create a source galaxy with a centered sersic light profile
    intensity=0.3,
    effective_radius=0.1,
    sersic_index=1,
)

source_galaxy = al.Galaxy(redshift=1.0, **source_dict)


lens_NFW_dict = {}
lens_NFW_dict["subhalo"] = al.mp.NFWTruncatedMCRLudlowSph(
    centre=(0, 0), mass_at_200=10**12
)
lens_NFW_galaxy = al.Galaxy(redshift=0.5, **lens_NFW_dict)

tracer_1 = al.Tracer.from_galaxies(
    galaxies=[lens_NFW_galaxy, source_galaxy]
)  # Create a lens galaxy with a NFW profile of mass 10^12 centered at 0,0
image_1 = simulator.via_tracer_from(tracer_1, grid_2d)

plotter_1 = al.plot.Array2DPlotter(array=image_1.data)
plotter_1.figure_2d()


lens_SIS_dict = {}
lens_SIS_dict["mass"] = al.mp.IsothermalSph(centre=(0, 0), einstein_radius=2.0)
lens_SIS_galaxy = al.Galaxy(redshift=0.5, **lens_SIS_dict)

tracer_2 = al.Tracer.from_galaxies(
    galaxies=[lens_SIS_galaxy, source_galaxy]
)  # Create a lens galaxy with a SIS profile of centered at 0,0

image_2 = simulator.via_tracer_from(tracer_2, grid_2d)

plotter_2 = al.plot.Array2DPlotter(array=image_2.data)
plotter_2.figure_2d()


# It was Simon, my partner, and I's understanding that the above two would produce similar images but note the lack of any visible lensing in the image with only the NFW


lens_combined_1_dict = {}
lens_combined_1_dict["mass"] = al.mp.IsothermalSph(centre=(0, 0), einstein_radius=2.0)
lens_combined_1_dict["subhalo"] = al.mp.NFWTruncatedMCRLudlowSph(
    centre=(-2.0, 0), mass_at_200=10**12
)

lens_combined_1_galaxy = al.Galaxy(redshift=0.5, **lens_combined_1_dict)

tracer_3 = al.Tracer.from_galaxies(
    galaxies=[lens_combined_1_galaxy, source_galaxy]
)  # When combining the two profiles togethetr some small perturbation is visibile but it's not as pronounced as I would have thought for such a massive halo

image_3 = simulator.via_tracer_from(tracer_3, grid_2d)

plotter_3 = al.plot.Array2DPlotter(array=image_3.data)
plotter_3.figure_2d()


lens_combined_2_dict = {}
lens_combined_2_dict["mass"] = al.mp.IsothermalSph(centre=(0, 0), einstein_radius=2.0)
lens_combined_2_dict["subhalo"] = al.mp.IsothermalSph(
    centre=(-2.0, 0), einstein_radius=0.2
)

lens_combined_2_galaxy = al.Galaxy(redshift=0.5, **lens_combined_2_dict)

tracer_4 = al.Tracer.from_galaxies(
    galaxies=[lens_combined_2_galaxy, source_galaxy]
)  # If I instead use a smaller SIS profile instead of a NFW profile I get a much more pronoucned effect on the Einstein ring

image_4 = simulator.via_tracer_from(tracer_4, grid_2d)

plotter_4 = al.plot.Array2DPlotter(array=image_4.data)
plotter_4.figure_2d()
