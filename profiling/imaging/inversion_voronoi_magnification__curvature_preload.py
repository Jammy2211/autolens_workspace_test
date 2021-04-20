"""
__PROFILING: Inversion VoronoiMagnification__

This profiling script times how long it takes to fit `Imaging` data with a `VoronoiMagnification` pixelization for
datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoLens** log likelihood function.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import time
from autoarray.inversion import mappers
import autolens as al

"""
The path all profiling results are output.
"""
file_path = os.path.join(
    "profiling", "times", al.__version__, "inversion_voronoi_magnification"
)

"""
The number of repeats used to estimate the `Inversion` run time.
"""
repeats = 1
print("Number of repeats = " + str(repeats))
print()

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 4
mask_radius = 3.5
psf_shape_2d = (21, 21)
pixelization_shape_2d = (57, 57)

print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"psf shape = {psf_shape_2d}")
print(f"pixelization shape = {pixelization_shape_2d}")

"""
The lens galaxy used to fit the data, which is identical to the lens galaxy used to simulate the data. 
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=4.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp.EllExponential(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=2.0,
        effective_radius=1.6,
    ),
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=45.0),
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.001, 0.001)),
)

"""
The source galaxy whose `VoronoiMagnification` `Pixelization` fits the data.
"""
pixelization = al.pix.VoronoiMagnification(shape=pixelization_shape_2d)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
    regularization=al.reg.Constant(coefficient=1.0),
)

"""
The simulated data comes at five resolution corresponding to five telescopes:

vro: pixel_scale = 0.2", fastest run times.
euclid: pixel_scale = 0.1", fast run times
hst: pixel_scale = 0.05", normal run times, represents the type of data we do most our fitting on currently.
hst_up: pixel_scale = 0.03", slow run times.
ao: pixel_scale = 0.01", very slow :(
"""
# instrument = "vro"
# instrument = "euclid"
instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

pixel_scales_dict = {"vro": 0.2, "euclid": 0.1, "hst": 0.05, "hst_up": 0.03, "ao": 0.01}
pixel_scale = pixel_scales_dict[instrument]

"""
Load the dataset for this instrument / resolution.
"""
dataset_path = path.join("dataset", "imaging", instrument)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scale,
)

"""
Apply the 2D mask, which for the settings above is representative of the masks we typically use to model strong lenses.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    sub_size=sub_size,
    radius=mask_radius,
)

masked_imaging = imaging.apply_mask(mask=mask)
masked_imaging = masked_imaging.apply_settings(
    settings=al.SettingsImaging(sub_size=sub_size)
)

"""
__Fit__

Performs the complete fit for the overall run-time to fit the lens model to the data.

This also ensures any uncached numba methods are called before profiling begins, and therefore compilation time
is not factored into the project run times.

https://github.com/Jammy2211/PyAutoLens/blob/master/autolens/fit/fit.py
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(imaging=masked_imaging, tracer=tracer)
fit.log_evidence


"""
__Lens Light (Grid2D)__

Compute the light profile of the foreground lens galaxy, which for this script uses an `EllpiticalSersic` bulge and
`EllExponential` disk. This computes the `image` of each `LightProfile` and adds them together. 

It also includes a `blurring_image` which represents all flux values not within the mask, but which will blur into the
mask after PSF convolution.

The calculation below uses a `Grid2D` object with a fixed sub-size of 2.

To see examples of `LightProfile` image calculations checkout the `image_2d_from_grid` methods at the following link:

https://github.com/Jammy2211/PyAutoGalaxy/blob/master/autogalaxy/profiles/light_profiles.py
"""
image = lens_galaxy.image_2d_from_grid(grid=masked_imaging.grid)
blurring_image = lens_galaxy.image_2d_from_grid(grid=masked_imaging.blurring_grid)

"""
__Lens Light (Grid2DIterate)__

This is an alternative method of computing the lens galaxy images above, which uses a grid whose sub-size adaptively
increases depending on a required fractional accuracy of the light profile.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/grid_2d_iterate.py
"""
masked_imaging_iterate = imaging.apply_mask(mask=mask)
masked_imaging_iterate = masked_imaging_iterate.apply_settings(
    settings=al.SettingsImaging(grid_class=al.Grid2DIterate)
)

image = lens_galaxy.image_2d_from_grid(grid=masked_imaging_iterate.grid)
blurring_image = lens_galaxy.image_2d_from_grid(grid=masked_imaging.blurring_grid)

"""
__Lens Light Convolution__

Convolves the lens light image above with the PSF and subtracts this from the observed image.

This uses the methods in `Convolver.__init__` and `Convolver.convolve_image`:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/operators/convolver.py
"""
convolved_image = masked_imaging.convolver.convolve_image(
    image=image, blurring_image=blurring_image
)

"""
__Ray Tracing (SIE)__

Compute the deflection angles and ray-trace the image-pixels to the source plane. The run-time of this step depends
on the lens galaxy mass model, for this example we use a fast `EllIsothermal` meaning this step is not the 
bottleneck.

This uses the fast `EllIsothermal` profile.

Deflection angle calculations are profiled fully in the package`profiling/deflections`.

To see examples of deflection angle calculations checkout the `deflections_2d_from_grid` methods at the following link:

https://github.com/Jammy2211/PyAutoGalaxy/blob/master/autogalaxy/profiles/mass_profiles/total_mass_profiles.py

Ray tracing is handled in the following module:

https://github.com/Jammy2211/PyAutoLens/blob/master/autolens/lens/ray_tracing.py

The image-plane pixelization computed below must be ray-traced just like the image-grid and is therefore included in
the profiling time below.
"""
sparse_image_plane_grid = al.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
    grid=masked_imaging.grid, unmasked_sparse_shape=pixelization.shape
)

tracer.deflections_2d_from_grid(grid=sparse_image_plane_grid)
traced_grid = tracer.traced_grids_of_planes_from_grid(grid=masked_imaging.grid)[-1]

"""
__Image-plane Pixelization (Gridding)__

The `VoronoiMagnification` begins by determining what will become its the source-pixel centres by calculating them 
in the image-plane. 

This calculation is performed by overlaying a uniform regular grid with `pixelization_shape_2d` over the image-plane 
mask and retaining all pixels that fall within the mask. This grid is called a `Grid2DSparse` as it retains information
on the mapping between the sparse image-plane pixelization grid and full resolution image grid.

Checkout the functions `Grid2DSparse.__init__` and `Grid2DSparse.from_grid_and_unmasked_2d_grid_shape`

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/grids.py 
"""
sparse_image_plane_grid = al.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
    grid=masked_imaging.grid, unmasked_sparse_shape=pixelization.shape
)

traced_sparse_grid = tracer.traced_sparse_grids_of_planes_from_grid(
    grid=masked_imaging.grid
)[-1]

"""
__Border Relocation__

Coordinates that are ray-traced near the `MassProfile` centre are heavily demagnified and may trace to far outskirts of
the source-plane. We relocate these pixels to the edge of the source-plane border (defined via the border of the 
image-plane mask) have as described in **HowToLens** chapter 4 tutorial 5. 

Checkout the function `relocated_grid_from_grid` for a full description of the method:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/abstract_grid_2d.py
"""
relocated_grid = traced_grid.relocated_grid_from_grid(grid=traced_grid)

"""
__Border Relocation Pixelization__

The pixelization grid is also subject to border relocation.

Checkout the function `relocated_pxielization_grid_from_pixelization_grid` for a full description of the method:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/abstract_grid_2d.py
"""
relocated_pixelization_grid = traced_grid.relocated_pixelization_grid_from_pixelization_grid(
    pixelization_grid=traced_sparse_grid
)

"""
__Voronoi Mesh__

The relocated pixelization grid is now used to create the `Pixelization`'s Voronoi grid using the scipy.spatial library.

The array `sparse_index_for_slim_index` encodes the closest source pixel of every pixel on the (full resolution)
sub image-plane grid. This is used for efficiently pairing every image-plane pixel to its corresponding source-plane
pixel.

Checkout `Grid2DVoronoi.__init__` for a full description:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/grid_2d_pixelization.py
"""
grid_voronoi = al.Grid2DVoronoi(
    grid=relocated_pixelization_grid,
    nearest_pixelization_index_for_slim_index=sparse_image_plane_grid.sparse_index_for_slim_index,
)

"""
We now combine grids computed above to create a `Mapper`, which describes how every image-plane (sub-)pixel maps to
every source-plane Voronoi pixel. 

There are two computationally steps in this calculation, which we profile individually below. Therefore, we do not
time the calculation below, but will use the `mapper` that comes out later in the profiling script.

Checkout the modules below for a full description of a `Mapper` and the `mapping_matrix`:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/mappers.py
https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/util/mapper_util.py 
"""
mapper = mappers.MapperVoronoi(
    source_grid_slim=relocated_grid,
    source_pixelization_grid=grid_voronoi,
    data_pixelization_grid=sparse_image_plane_grid,
)

"""
__Image-Source Pairing__

The `Mapper` contains:

 1) The traced grid of (y,x) source pixel coordinate centres.
 2) The traced grid of (y,x) image pixel coordinates.
 
The function below pairs every image-pixel coordinate to every source-pixel centre.

In the API, the `pixelization_index` refers to the source pixel index (e.g. source pixel 0, 1, 2 etc.) whereas the 
sub_slim index refers to the index of a sub-gridded image pixel (e.g. sub pixel 0, 1, 2 etc.). The docstrings of the
function below describes this method.

VoronoiMapper.pixelization_index_for_sub_slim_index:

 https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/mappers.py
 
pixelization_index_for_voronoi_sub_slim_index_from:
 
 https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/util/mapper_util.py 
"""
pixelization_index_for_sub_slim_index = mapper.pixelization_index_for_sub_slim_index

"""
__Mapping Matrix (f)__

The `mapping_matrix` is a matrix that represents the image-pixel to source-pixel mappings above in a 2D matrix. It is
described at the GitHub link below and in ther following paper as matrix `f` https://arxiv.org/pdf/astro-ph/0302587.pdf.

Mapper.__init__:

 https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/mappers.py
"""
mapping_matrix = al.util.mapper.mapping_matrix_from(
    pixelization_index_for_sub_slim_index=pixelization_index_for_sub_slim_index,
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_grid_slim.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper._slim_index_for_sub_slim_index,
    sub_fraction=mapper.source_grid_slim.mask.sub_fraction,
)


"""
__Blurred Mapping Matrix (f_blur)__

For a given source pixel on the mapping matrix, we can use it to map it to a set of image-pixels in the image plane.
This therefore creates a 'image' of the source pixel (which corresponds to a set of values that mostly zeros, but with
1's where mappings occur).

Before reconstructing the source, we blur every one of these source pixel images with the Point Spread Function of our 
dataset via 2D convolution. This uses the methods in `Convolver.__init__` and `Convolver.convolve_mapping_matrix`:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/operators/convolver.py
"""
blurred_mapping_matrix = masked_imaging.convolver.convolve_mapping_matrix(
    mapping_matrix=mapping_matrix
)

"""
__Data Vector (D)__

To solve for the source pixel fluxes we now pose the problem as a linear inversion which we use the NumPy linear 
algebra libraries to solve. The linear algebra is based on the paper https://arxiv.org/pdf/astro-ph/0302587.pdf .

This requires us to convert the blurred mapping matrix and our data / noise map into matrices of certain dimensions. 

The `data_vector` D is the first such matrix, which is given by equation (4) 
in https://arxiv.org/pdf/astro-ph/0302587.pdf. 

The calculation is performed by thge method `data_vector_via_blurred_mapping_matrix_from` at:

 https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/util/inversion_util.py
"""
subtracted_image = masked_imaging.image - convolved_image
data_vector = al.util.inversion.data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix=blurred_mapping_matrix,
    image=subtracted_image,
    noise_map=masked_imaging.noise_map,
)

"""
__Curvature Matrix (F) No Preload__

The `curvature_matrix` F is the second matrix, given by equation (4) in https://arxiv.org/pdf/astro-ph/0302587.pdf.

The calculation is performed by the method `curvature_matrix_via_mapping_matrix_from` at:

 https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/util/inversion_util.py
"""
start = time.time()
for i in range(repeats):
    curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, noise_map=masked_imaging.noise_map
    )
print(f"Curvature Matrix (F) = {(time.time() - start) / repeats}")


"""
__Curvature Matrix (F) via Sparse Preload__

The `curvature_matrix` F is the second matrix, given by equation (4) in https://arxiv.org/pdf/astro-ph/0302587.pdf.

The calculation is performed by the method `curvature_matrix_via_mapping_matrix_from` at:

 https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/util/inversion_util.py
"""
start = time.time()
curvature_matrix_sparse_preload, preload_counts = al.util.inversion.curvature_matrix_sparse_preload_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix
)
print(f"Curvature Matrix (F) Preloading time = {(time.time() - start) / repeats}")

curvature_matrix = al.util.inversion.curvature_matrix_via_sparse_preload_from(
    mapping_matrix=blurred_mapping_matrix,
    noise_map=masked_imaging.noise_map,
    curvature_matrix_sparse_preload=curvature_matrix_sparse_preload.astype("int"),
    curvature_matrix_preload_counts=preload_counts.astype("int"),
)

start = time.time()
for i in range(repeats):
    curvature_matrix = al.util.inversion.curvature_matrix_via_sparse_preload_from(
        mapping_matrix=blurred_mapping_matrix,
        noise_map=masked_imaging.noise_map,
        curvature_matrix_sparse_preload=curvature_matrix_sparse_preload.astype("int"),
        curvature_matrix_preload_counts=preload_counts.astype("int"),
    )
print(f"Curvature Matrix (F) via Sparse Preload = {(time.time() - start) / repeats}")
