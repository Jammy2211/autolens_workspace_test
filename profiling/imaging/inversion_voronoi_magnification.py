import time
from os import path
import numpy as np

from autoarray.structures import grids
import autolens as al
import autolens.plot as aplt

"""
Number of times the function is repeated for profiling.
"""
repeats = 1
print("Number of repeats = " + str(repeats))
print()

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 4
mask_radius = 3.0
psf_shape_2d = (21, 21)
pixelization_shape_2d = (30, 30)

print("sub grid size = " + str(sub_size))
print("circular mask mask_radius = " + str(mask_radius) + "\n")
print("psf shape = " + str(psf_shape_2d) + "\n")
print("pixelization shape = " + str(pixelization_shape_2d) + "\n")

"""
The lens galaxy used to fit the data, which is identical to the lens galaxy used to simulate the data. 

It may be beneficial to profile the code for cases where the lens model is inaccurate.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=45.0),
    ),
)

"""
The pixelization + source galaxy which fits the data, which in this script is a Voronoi mesh adapted to the lens model
magnification.
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
dataset_path = path.join("dataset", "instruments", instrument)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scale,
)

"""
Apply a 2D mask, which is roughly representative of the masks we typically use to model data.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    sub_size=sub_size,
    radius=mask_radius,
)

masked_imaging = al.MaskedImaging(
    imaging=imaging, mask=mask, settings=al.SettingsMaskedImaging(sub_size=sub_size)
)

print(f"Inversion fit run times for image type {instrument} \n")

### These two numbers are the primary driver of run time. More pixels = longer run time.

print(f"Number of pixels = {masked_imaging.grid.shape_slim} \n")
print(f"Number of sub-pixels = {masked_imaging.grid.sub_shape_slim} \n")


start_overall = time.time()

"""
This step computes the deflection angles and ray-tracing the image-pixels to the source plane. It isn't the bottleneck
we are currently concerned with, and there is a dedicated profiling script for deflection angles.

To see examples of deflection angle calculations checkout the `deflections_from_grid` methods at the following link:

https://github.com/Jammy2211/PyAutoGalaxy/blob/master/autogalaxy/profiles/mass_profiles/total_mass_profiles.py
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

start = time.time()
for i in range(repeats):
    traced_grid = tracer.traced_grids_of_planes_from_grid(grid=masked_imaging.grid)[-1]
diff = time.time() - start
print("Time to Setup Traced Grid2D = {}".format(diff / repeats))

"""
The `VoronoiMagnification` requires us to extract a sparse set of image pixels that will act as the centres of the
source-pixels in the source plane.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/grids.py

Checkout: 

 Grid2DSparse.__init__
 Grid2DSparse.from_grid_and_unmasked_2d_grid_shape
"""
for i in range(repeats):
    sparse_grid = grids.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
        grid=masked_imaging.grid, unmasked_sparse_shape=pixelization.shape
    )
diff = time.time() - start
print("Time to create sparse grid = {}".format(diff / repeats))

"""
This step is the same as above, but using a `sparse` grid of image pixels which defines the centres of the 
source-pixelization. In terms of optimization it isn't too important.
"""
start = time.time()
for i in range(repeats):
    traced_sparse_grid = tracer.traced_sparse_grids_of_planes_from_grid(
        grid=masked_imaging.grid
    )[-1]

diff = time.time() - start
print("Time to Setup Traced Sparse Grid2D = {}".format(diff / repeats))

"""
To create a `Mapper` we have to relocate pixels that go over the border to the border (chapter 4, tutorial 5). The 
function below times how long the border relocation takes.

Checkout the function `relocated_grid_from_grid` for a full description of the method:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/abstract_grid.py
"""

start = time.time()
for i in range(repeats):
    relocated_grid = traced_grid.relocated_grid_from_grid(grid=traced_grid)
    relocated_pixelization_grid = traced_grid.relocated_pixelization_grid_from_pixelization_grid(
        pixelization_grid=traced_sparse_grid
    )
diff = time.time() - start
print("Time to perform border relocation = {}".format(diff / repeats))

"""
The creation of a mapper also creates a Voronoi grid using the scipy.spatial library, which is profiled below.

Checkout `GridVoronoi.__init__` for a full description:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/pixelization.py
"""
start = time.time()
for i in range(repeats):
    grid_voronoi = al.Grid2DVoronoi(grid=relocated_pixelization_grid)
diff = time.time() - start
print("Time to create Voronoi mesh = {}".format(diff / repeats))

"""
The profiling steps below need a `Mapper` object, which I make below. This performs the same steps as above 
(border relocation, Voronoi mesh) so I havene't bothered timing it.
"""
mapper = pixelization.mapper_from_grid_and_sparse_grid(
    grid=traced_grid,
    sparse_grid=traced_sparse_grid,
    settings=al.SettingsPixelization(use_border=True),
)

"""
The `Mapper` contains:

 1) The traced grid of (y,x) source pixel coordinate centres.
 2) The traced grid of (y,x) image pixel coordinates.
 
The function below pairs every image-pixel coordinate to every source-pixel centre.

In the API, the `pixelization_index` refers to the source pixel index (e.g. source pixel 0, 1, 2 etc.) whereas the 
sub_slim index refers to the index of a sub-gridded image pixel (e.g. sub pixel 0, 1, 2 etc.). The docstrings of the
function below describes this in a bit more detail (this code is quite messy, we may want to discuss this on Zoom).

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/mappers.py -> VoronoiMapper.pixelization_index_for_sub_slim_index
https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/util/mapper_util.py -> pixelization_index_for_voronoi_sub_slim_index_from
"""
start = time.time()
for i in range(repeats):
    pixelization_index_for_sub_slim_index = mapper.pixelization_index_for_sub_slim_index
diff = time.time() - start
print(
    "Time to pair every sub-image pixel to every source pixel = {}".format(
        diff / repeats
    )
)


"""
The mapping matrix is a matrix that represents the image-pixel to source-pixel mappings above in a 2D matrix. It is
described at the GitHub link below.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/mappers.py -> Mapper.__init__
"""
start = time.time()
for i in range(repeats):
    mapping_matrix = mapper.mapping_matrix

    al.util.mapper.mapping_matrix_from(
        pixelization_index_for_sub_slim_index=pixelization_index_for_sub_slim_index,
        pixels=mapper.pixels,
        total_mask_pixels=mapper.source_grid_slim.mask.pixels_in_mask,
        slim_index_for_sub_slim_index=mapper._slim_index_for_sub_slim_index,
        sub_fraction=mapper.source_grid_slim.mask.sub_fraction,
    )

diff = time.time() - start
print("Time to compute mapping matrix = {}".format(diff / repeats))

"""
For a given source pixel on the mapping matrix, we can use it to map it to a set of image-pixels in the image plane.
This therefore creates a 'image' of the source pixel (which corresponds to a set of values that mostly zeros, but with
1's where mappings occur).

We now blur every source pixel image with the Point Spread Function of our dataset, which uses a 2D convolution.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/operators/convolver.py

Checkout:
 
Convolver.__init__
Convolver.convolve_mapping_matrix
"""
start = time.time()
for i in range(repeats):
    blurred_mapping_matrix = masked_imaging.convolver.convolve_mapping_matrix(
        mapping_matrix=mapping_matrix
    )
diff = time.time() - start
print("Time to compute blurred mapping matrix = {}".format(diff / repeats))

"""
To solve for the source pixel fluxes we now pose the problem as a linear inversion which we use the NumPy linear 
algebra libraries to solve. The linear algebra is based on the following paper, feel free to check it out but it
will probably easier I describe these steps on Zoom: 

 https://arxiv.org/pdf/astro-ph/0302587.pdf

This requires us to convert the blurred mapping matrix and our data / noise map into 
matrices of certain dimensions.

The `data_vector` D is the first such matrix.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/util/inversion_util.py
"""
start = time.time()
for i in range(repeats):
    data_vector = al.util.inversion.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix,
        image=masked_imaging.image,
        noise_map=masked_imaging.noise_map,
    )
diff = time.time() - start
print("Time to compute data vector = {}".format(diff / repeats))

"""
The `curvature_matrix` F is the second matrix.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/util/inversion_util.py
"""
start = time.time()
for i in range(repeats):
    curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, noise_map=masked_imaging.noise_map
    )
diff = time.time() - start
print("Time to compute curvature matrix = {}".format(diff / repeats))

"""
The regularization matrix H is used to impose smoothness on our source reconstruction. This enters the linear algebra
system we solve for using D and F above.

A complete descritpion of regularization is at the link below.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/regularization.py
"""
start = time.time()
for i in range(repeats):
    regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
        coefficient=1.0,
        pixel_neighbors=mapper.source_pixelization_grid.pixel_neighbors,
        pixel_neighbors_size=mapper.source_pixelization_grid.pixel_neighbors_size,
    )
diff = time.time() - start
print("Time to compute reguarization matrix = {}".format(diff / repeats))

"""
The linear system of equations solves for F + regularization_coefficient*H.
"""
start = time.time()
for i in range(repeats):
    curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)
diff = time.time() - start
print("Time to compute curvature reguarization Matrix = {}".format(diff / repeats))

"""
This solves the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D. 

S is the vector of reconstructed source fluxes.
"""
start = time.time()
for i in range(repeats):
    reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)
diff = time.time() - start
print(
    "Time to solve linear system and perform reconstruction = {}".format(diff / repeats)
)

"""
The log determinant of [F + reg_coeff*H] is something we need to compute.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversions.py
"""
start = time.time()
for i in range(repeats):
    2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(curvature_reg_matrix))))
diff = time.time() - start
print("Time to compute log(det(F + reg_coeffH)) = {}".format(diff / repeats))

"""
We also need log determinant of H.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversions.py
"""
start = time.time()
for i in range(repeats):
    2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(regularization_matrix))))
diff = time.time() - start
print("Time to compute log(det(H)) = {}".format(diff / repeats))

"""
Finally, now we have the reconstructed source pixel fluxes we can map the source flux back to the image plane (via
the mapping_matrix) and reconstruct the image data.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversions.py
"""
start = time.time()
for i in range(repeats):
    al.util.inversion.mapped_reconstructed_data_from(
        mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
    )
diff = time.time() - start
print("Time to compute mapped reconstruction = {}".format(diff / repeats))

"""
This performs the complete fit, and is thus the overall run-time taken to fit the lens model to the data (excluding the
deflection angle calculations).

https://github.com/Jammy2211/PyAutoLens/blob/master/autolens/fit/fit.py
"""
start = time.time()
for i in range(repeats):
    fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)
    fit.log_evidence
diff = time.time() - start
print("Time to perform complete fit = {}".format(diff / repeats))


"""
Here is a pretty picture.
"""

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)
fit_imaging_plotter.subplot_fit_imaging()
fit_imaging_plotter.subplot_of_planes(plane_index=1)
