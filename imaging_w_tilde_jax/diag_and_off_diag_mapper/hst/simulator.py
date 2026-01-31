from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/image.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/psf.fits`.
"""
dataset_type = "imaging"
dataset_name = "simple"

"""
The path where the dataset will be output. 

In this example, this is: `/autolens_workspace/dataset/imaging/simple`
"""
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__
"""
grid = al.Grid2D.uniform(
    shape_native=(300, 300),
    pixel_scales=0.05,
)

"""
__Over Sampling__
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

psf = al.Kernel2D.from_gaussian(
    shape_native=(21, 21), sigma=0.1, pixel_scales=grid.pixel_scales
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Ray Tracing__

"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=0.3,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

source_galaxy_1 = al.Galaxy(
    redshift=2.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)


"""
We now pass these galaxies to a `Tracer`, which performs the ray-tracing calculations they describe and returns
the image of the strong lens system they produce.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

"""
By passing the `Tracer` and grid to the simulator, we create the simulated CCD imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

dataset.output_to_fits(
    data_path="data.fits",
    psf_path="psf.fits",
    noise_map_path="noise_map.fits",
    overwrite=True,
)


mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=mask_radius
)

dataset = dataset.apply_mask(mask=mask)

dataset = dataset.apply_over_sampling(over_sample_size_pixelization=4)

image_mesh = None
mesh_shape = (30, 30)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

source_pixel_zeroed_indices = al.util.mesh.rectangular_edge_pixel_list_from(
    mesh_shape
)

preloads = al.Preloads(
    mapper_indices=al.mapper_indices_from(
        total_linear_light_profiles=0,
        total_mapper_pixels=total_mapper_pixels*2
    ),
    source_pixel_zeroed_indices=source_pixel_zeroed_indices + list(total_mapper_pixels + np.array(source_pixel_zeroed_indices))
)

mesh = al.mesh.RectangularAdaptDensity(shape=mesh_shape)
regularization = al.reg.Constant(coefficient=1.0)

pixelization = al.Pixelization(
    mesh=mesh, regularization=regularization
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=0.3,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    pixelization=pixelization
)

mesh = al.mesh.RectangularAdaptDensity(shape=mesh_shape)
regularization = al.reg.Constant(coefficient=1.0)

pixelization = al.Pixelization(
    mesh=mesh, regularization=regularization
)

source_galaxy_1 = al.Galaxy(
    redshift=2.0,
    pixelization=pixelization
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

tracer_to_inversion = al.TracerToInversion(
    dataset=dataset,
    tracer=tracer,
    preloads=preloads,
)

mapper_0 = list(tracer_to_inversion.mapper_galaxy_dict.keys())[0]
mapper_1 = list(tracer_to_inversion.mapper_galaxy_dict.keys())[1]

np.save(file="pix_indexes_for_sub_slim_index_0", arr=mapper_0.pix_indexes_for_sub_slim_index)
np.save(file="pix_sizes_for_sub_slim_index_0", arr=mapper_0.pix_sizes_for_sub_slim_index)
np.save(file="pix_weights_for_sub_slim_index_0", arr=mapper_0.pix_weights_for_sub_slim_index)

np.save(file="pix_indexes_for_sub_slim_index_1", arr=mapper_1.pix_indexes_for_sub_slim_index)
np.save(file="pix_sizes_for_sub_slim_index_1", arr=mapper_1.pix_sizes_for_sub_slim_index)
np.save(file="pix_weights_for_sub_slim_index_1", arr=mapper_1.pix_weights_for_sub_slim_index)

np.save(file="native_index_for_slim_index", arr=dataset.grid.mask.derive_indexes.native_for_slim.astype("int"))
np.save(file="pix_pixels", arr=total_mapper_pixels)
np.save(file="curvature_matrix.npy", arr=tracer_to_inversion.inversion.curvature_matrix)
np.save(file="sub_fraction.npy", arr=dataset.grids.pixelization.over_sampler.sub_fraction)
np.save(file="slim_index_for_sub_slim_index.npy", arr=dataset.grids.pixelization.over_sampler.slim_for_sub_slim)

plotter = aplt.InversionPlotter(
    inversion=tracer_to_inversion.inversion,
    mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=".", format="png"))
)

plotter.set_filename("mapper_0")
plotter.subplot_of_mapper(mapper_index=0)

plotter.set_filename("mapper_1")
plotter.subplot_of_mapper(mapper_index=1)