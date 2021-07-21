"""
__PROFILING: Interferometer Voronoi Magnification Fit__

This profiling script times how long an `Inversion` takes to fit `Interferometer` data.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", ""))

import autolens as al
import autolens.plot as aplt
from autoarray.inversion import mappers
import json
import time
import numpy as np

"""
The path all profiling results are output.
"""
file_path = os.path.join(
    "interferometer", "profiling", "times", al.__version__, "inversion_voronoi_magnification"
)

"""
The number of repeats used to estimate the `Inversion` run time.
"""
repeats = 3

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 1
pixel_scales = (0.05, 0.05)
mask_radius = 3.5
pixelization_shape_2d = (57, 57)

print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"pixelization shape = {pixelization_shape_2d}")

"""
These settings control the run-time of the `Inversion` performed on the `Interferometer` data.
"""
transformer_class = al.TransformerDFT
use_linear_operators = False

"""
Set up the `Interferometer` dataset we fit. This includes the `real_space_mask` that the source galaxy's 
`Inversion` is evaluated using via mapping to Fourier space using the `Transformer`.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=pixel_scales, sub_size=sub_size, radius=mask_radius
)

"""
Load the strong lens dataset `mass_sie__source_sersic` `from .fits files.
"""
instrument = "sma"

dataset_path = path.join("dataset", "interferometer", "instruments", instrument)

interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

"""
Set up the lens and source galaxies used to profile the fit. The lens galaxy uses the true model, whereas the source
galaxy includes the `Pixelization` and `Regularization` we profile.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=45.0),
    ),
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


interferometer = interferometer.apply_settings(
    settings=al.SettingsInterferometer(transformer_class=transformer_class)
)

"""
Tracers using a power-law and decomposed mass model, just to provide run times of mode complex mass models.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllPowerLaw(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=45.0),
        slope=2.0
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.001, 0.001)),
)
tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy]
)

"""
__Fit__

Performs the complete fit for the overall run-time to fit the lens model to the data.

This also ensures any uncached numba methods are called before profiling begins, and therefore compilation time
is not factored into the project run times.

https://github.com/Jammy2211/PyAutoLens/blob/master/autolens/fit/fit.py
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitInterferometer(
    interferometer=interferometer,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(
        use_linear_operators=use_linear_operators
    ),
)
fit.log_evidence

start = time.time()
for i in range(repeats):
    fit = al.FitInterferometer(
        interferometer=interferometer,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(
            use_linear_operators=use_linear_operators
        ),
    )
    fit.log_evidence

fit_time = (time.time() - start) / repeats

"""
The profiling dictionary stores the run time of every total mass profile.
"""
profiling_dict = {}

"""
We now start of the profiling timer and iterate through every step of the fitting strong lens data with 
a `VoronoiMagnification` pixelization. We provide a description of every step to give an overview of what is the reason
for its run time.
"""
start_overall = time.time()

"""
__Ray Tracing (Power-Law)__

Compute the deflection angles and ray-trace the image-pixels to the source plane. The run-time of this step depends
on the lens galaxy mass model, for this example we use a `EllPowerLaw`.

Deflection angle calculations are profiled fully in the package`profiling/deflections`.

To see examples of deflection angle calculations checkout the `deflections_2d_from_grid` methods at the following link:

https://github.com/Jammy2211/PyAutoGalaxy/blob/master/autogalaxy/profiles/mass_profiles/total_mass_profiles.py

Ray tracing is handled in the following module:

https://github.com/Jammy2211/PyAutoLens/blob/master/autolens/lens/ray_tracing.py

The image-plane pixelization computed below must be ray-traced just like the image-grid and is therefore included in
the profiling time below..
"""
sparse_image_plane_grid = al.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
    grid=interferometer.grid, unmasked_sparse_shape=pixelization.shape
)

start = time.time()
for i in range(repeats):
    tracer.deflections_2d_from_grid(grid=sparse_image_plane_grid)
    traced_grid = tracer.traced_grids_of_planes_from_grid(
        grid=interferometer.grid
    )[-1]

profiling_dict["Ray Tracing (Power-Law)"] = (time.time() - start) / repeats


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
start = time.time()
for i in range(repeats):
    sparse_image_plane_grid = al.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
        grid=interferometer.grid, unmasked_sparse_shape=pixelization.shape
    )

profiling_dict["Image-plane Pixelization (Gridding)"] = (time.time() - start) / repeats

traced_sparse_grid = tracer.traced_sparse_grids_of_planes_from_grid(
    grid=interferometer.grid
)[-1]

"""
__Border Relocation__

Coordinates that are ray-traced near the `MassProfile` centre are heavily demagnified and may trace to far outskirts of
the source-plane. We relocate these pixels to the edge of the source-plane border (defined via the border of the 
image-plane mask) have as described in **HowToLens** chapter 4 tutorial 5. 

Checkout the function `relocated_grid_from_grid` for a full description of the method:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/abstract_grid_2d.py
"""
start = time.time()
for i in range(repeats):
    relocated_grid = traced_grid.relocated_grid_from_grid(grid=traced_grid)
profiling_dict["Border Relocation"] = (time.time() - start) / repeats

"""
__Border Relocation Pixelization__

The pixelization grid is also subject to border relocation.

Checkout the function `relocated_pxielization_grid_from_pixelization_grid` for a full description of the method:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/abstract_grid_2d.py
"""
start = time.time()
for i in range(repeats):
    relocated_pixelization_grid = traced_grid.relocated_pixelization_grid_from_pixelization_grid(
        pixelization_grid=traced_sparse_grid
    )
profiling_dict["Border Relocation Pixelization"] = (time.time() - start) / repeats

"""
__Voronoi Mesh__

The relocated pixelization grid is now used to create the `Pixelization`'s Voronoi grid using the scipy.spatial library.

The array `sparse_index_for_slim_index` encodes the closest source pixel of every pixel on the (full resolution)
sub image-plane grid. This is used for efficiently pairing every image-plane pixel to its corresponding source-plane
pixel.

Checkout `Grid2DVoronoi.__init__` for a full description:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/grid_2d_pixelization.py
"""
start = time.time()
for i in range(repeats):
    grid_voronoi = al.Grid2DVoronoi(
        grid=relocated_pixelization_grid,
        nearest_pixelization_index_for_slim_index=sparse_image_plane_grid.sparse_index_for_slim_index,
    )
profiling_dict["Voronoi Mesh"] = (time.time() - start) / repeats

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
start = time.time()
for i in range(repeats):
    pixelization_index_for_sub_slim_index = mapper.pixelization_index_for_sub_slim_index
diff = (time.time() - start) / repeats
profiling_dict["Image-Source Pairing"] = (time.time() - start) / repeats

"""
__Mapping Matrix (f)__

The `mapping_matrix` is a matrix that represents the image-pixel to source-pixel mappings above in a 2D matrix. It is
described at the GitHub link below and in ther following paper as matrix `f` https://arxiv.org/pdf/astro-ph/0302587.pdf.

Mapper.__init__:

 https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/mappers.py
"""
start = time.time()
for i in range(repeats):
    mapping_matrix = al.util.mapper.mapping_matrix_from(
        pixelization_index_for_sub_slim_index=pixelization_index_for_sub_slim_index,
        pixels=mapper.pixels,
        total_mask_pixels=mapper.source_grid_slim.mask.pixels_in_mask,
        slim_index_for_sub_slim_index=mapper._slim_index_for_sub_slim_index,
        sub_fraction=mapper.source_grid_slim.mask.sub_fraction,
    )

profiling_dict["Mapping Matrix (f)"] = (time.time() - start) / repeats


"""
__Transformed Mapping Matrix (f_blur)__

For a given source pixel on the mapping matrix, we can use it to Fourier Transform (FT) it to a set of image plane 
visibilities. This therefore creates the visibilities of the source pixel (which corresponds to a fully dense set of 
non-zero values).

Before reconstructing the source, we FT every one of these source pixel images with the `uv_wavelengths` of our 
dataset. This uses the methods in `Transformer.__init__` and `Transformer.transform_mapping_matrix`:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/operators/transformer.py
"""
start = time.time()
for i in range(repeats):
    transformed_mapping_matrix = interferometer.transformer.transform_mapping_matrix(
        mapping_matrix=mapping_matrix
    )
profiling_dict["Transform Mapping Matrix (f_transform)"] = (time.time() - start) / repeats

"""
__Data Vector (D)__

To solve for the source pixel fluxes we now pose the problem as a linear inversion which we use the NumPy linear 
algebra libraries to solve. The linear algebra is based on the paper https://arxiv.org/pdf/astro-ph/0302587.pdf .

This requires us to convert the transformed mapping matrix and our data / noise map into matrices of certain dimensions. 

The `data_vector` D is the first such matrix, which is given by equation (4) 
in https://arxiv.org/pdf/astro-ph/0302587.pdf. 

The calculation is performed by thge method `data_vector_via_blurred_mapping_matrix_from` at:

 https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/util/inversion_util.py
"""
start = time.time()
for i in range(repeats):
    data_vector = al.util.inversion.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        visibilities=interferometer.visibilities,
        noise_map=interferometer.noise_map,
    )
profiling_dict["Data Vector (D)"] = (time.time() - start) / repeats


"""
__Curvature Matrix (F)__

The `curvature_matrix` F is the second matrix, given by equation (4) in https://arxiv.org/pdf/astro-ph/0302587.pdf.

The calculation is performed by the method `curvature_matrix_via_mapping_matrix_from` at:

 https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/util/inversion_util.py
"""
start = time.time()
for i in range(repeats):
    curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=transformed_mapping_matrix, noise_map=interferometer.noise_map
    )
profiling_dict["Curvature Matrix (F)"] = (time.time() - start) / repeats

"""
__Regularization Matrix (H)__

The regularization matrix H is used to impose smoothness on our source reconstruction. This enters the linear algebra
system we solve for using D and F above and is given by equation (12) in https://arxiv.org/pdf/astro-ph/0302587.pdf.

A complete descrition of regularization is at the link below.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/regularization.py
"""
start = time.time()
for i in range(repeats):
    regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
        coefficient=1.0,
        pixel_neighbors=mapper.source_pixelization_grid.pixel_neighbors,
        pixel_neighbors_size=mapper.source_pixelization_grid.pixel_neighbors_size,
    )
profiling_dict["Regularization Matrix (H)"] = (time.time() - start) / repeats

"""
__F + Lamdba H__

The linear system of equations solves for F + regularization_coefficient*H.
"""
start = time.time()
for i in range(repeats):
    curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)
profiling_dict["F + Lambda H"] = (time.time() - start) / repeats

"""
__Source Reconstruction (S)__

Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12) 
of https://arxiv.org/pdf/astro-ph/0302587.pdf 

S is the vector of reconstructed source fluxes.
"""
start = time.time()
for i in range(repeats):
    reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)
profiling_dict["Source Reconstruction (S)"] = (time.time() - start) / repeats

"""
__Log Det [F + Lambda H]__

The log determinant of [F + reg_coeff*H] is used to determine the Bayesian evidence of the solution.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversions.py
"""
# start = time.time()
# for i in range(repeats):
#     2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(curvature_reg_matrix))))
# profiling_dict["Log Det [F + Lambda H]"] = (time.time() - start) / repeats

"""
__Log Det [Lambda H]__

The evidence also uses the log determinant of Lambda H.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversions.py
"""
start = time.time()
for i in range(repeats):
    2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(regularization_matrix))))
profiling_dict["Log Det [Lambda H]"] = (time.time() - start) / repeats

"""
__Visibility Reconstruction__

Finally, now we have the reconstructed source pixel fluxes we can map the source flux back to the image plane (via
the blurred mapping_matrix) and reconstruct the image data.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversions.py
"""
start = time.time()
for i in range(repeats):
    al.util.inversion.mapped_reconstructed_visibilities_from(
        transformed_mapping_matrix=transformed_mapping_matrix, reconstruction=reconstruction
    )
profiling_dict["Visibility Reconstruction"] = (time.time() - start) / repeats

"""
__Results__

These two numbers are the primary driver of run time. More pixels = longer run time.
"""

print(f"Inversion fit run times for image type {instrument} \n")
print(f"Number of pixels = {interferometer.grid.shape_slim} \n")
print(f"Number of sub-pixels = {interferometer.grid.sub_shape_slim} \n")

"""
Print the profiling results of every step of the fit for command line output when running profiling scripts.
"""
for key, value in profiling_dict.items():
    print(key, value)

"""
__Output__

Output the profiling run times as a dictionary so they can be used in `profiling/graphs.py` to create graphs of the
profile run times.

This is stored in a folder using the **PyAutoLens** version number so that profiling run times can be tracked through
**PyAutoLens** development.
"""
if not os.path.exists(file_path):
    os.makedirs(file_path)

filename = f"{instrument}_profiling_dict.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(profiling_dict, outfile)

"""
Output the profiling run time of the entire fit.
"""
filename = f"{instrument}_fit_time.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(fit_time, outfile)

"""
Output an image of the fit, so that we can inspect that it fits the data as expected.
"""
mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_fit_interferometer", format="png"
    )
)
fit_interferometer_plotter = aplt.FitInterferometerPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_interferometer_plotter.subplot_fit_interferometer()
#fit_interferometer_plotter.subplot_fit_dirty_images()
#fit_interferometer_plotter.subplot_fit_real_space()

"""
The `info_dict` contains all the key information of the analysis which describes its run times.
"""
info_dict = {}
info_dict["repeats"] = repeats
info_dict["image_pixels"] = interferometer.grid.sub_shape_slim
info_dict["sub_size"] = sub_size
info_dict["mask_radius"] = mask_radius
# info_dict["source_pixels"] = len(reconstruction)

print(info_dict)

with open(path.join(file_path, f"{instrument}_info.json"), "w") as outfile:
    json.dump(info_dict, outfile)