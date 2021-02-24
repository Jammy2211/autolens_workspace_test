"""
__PROFILING: Inversion VoronoiMagnification__

This profiling script times how long it takes to fit `Imaging` data with a `VoronoiMagnification` pixelization for
datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoLens** log likelihood function.
"""
import time
from os import path
import numpy as np
import os
import json
from autoarray.inversion import mappers
import autolens as al
import autolens.plot as aplt

"""
The path all profiling results are output.
"""
file_path = os.path.join(
    "profiling", "times", al.__version__, "inversion_voronoi_magnification_high_reg"
)

"""
The number of repeats used to estimate the `Inversion` run time.
"""
repeats = 10
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
    bulge=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, phi=45.0),
        intensity=4.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp.EllipticalExponential(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, phi=30.0),
        intensity=2.0,
        effective_radius=1.6,
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=45.0),
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
    regularization=al.reg.Constant(coefficient=1e4),
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

masked_imaging = al.MaskedImaging(
    imaging=imaging, mask=mask, settings=al.SettingsMaskedImaging(sub_size=sub_size)
)

"""
Tracers using a power-law and decomposed mass model, just to provide run times of mode complex mass models.
"""
lens_galaxy_power_law = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalPowerLaw(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=45.0),
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.001, 0.001)),
)
tracer_power_law = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy_power_law, source_galaxy]
)

lens_galaxy_decomposed = al.Galaxy(
    redshift=0.5,
    bulge=al.lmp.EllipticalSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, phi=45.0),
        intensity=4.0,
        effective_radius=0.6,
        sersic_index=3.0,
        mass_to_light_ratio=0.05,
    ),
    disk=al.lmp.EllipticalExponential(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, phi=30.0),
        intensity=2.0,
        effective_radius=1.6,
        mass_to_light_ratio=0.05,
    ),
    dark=al.mp.EllipticalNFW(
        centre=(0.0, 0.0),
        elliptical_comps=(0.05, 0.05),
        kappa_s=0.12,
        scale_radius=20.0,
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.001, 0.001)),
)
tracer_decomposed = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy_decomposed, source_galaxy]
)

"""
__Fit__

Performs the complete fit for the overall run-time to fit the lens model to the data.

This also ensures any uncached numba methods are called before profiling begins, and therefore compilation time
is not factored into the project run times.

https://github.com/Jammy2211/PyAutoLens/blob/master/autolens/fit/fit.py
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)
fit.log_evidence

start = time.time()
for i in range(repeats):
    fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)
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
__Lens Light (Grid2D)__

Compute the light profile of the foreground lens galaxy, which for this script uses an `EllpiticalSersic` bulge and
`EllipticalExponential` disk. This computes the `image` of each `LightProfile` and adds them together. 

It also includes a `blurring_image` which represents all flux values not within the mask, but which will blur into the
mask after PSF convolution.

The calculation below uses a `Grid2D` object with a fixed sub-size of 2.

To see examples of `LightProfile` image calculations checkout the `image_from_grid` methods at the following link:

https://github.com/Jammy2211/PyAutoGalaxy/blob/master/autogalaxy/profiles/light_profiles.py
"""
start = time.time()
for i in range(repeats):
    image = lens_galaxy.image_from_grid(grid=masked_imaging.grid)
    blurring_image = lens_galaxy.image_from_grid(grid=masked_imaging.blurring_grid)

profiling_dict["Lens Light (Grid2D)"] = (time.time() - start) / repeats

"""
__Lens Light (Grid2DIterate)__

This is an alternative method of computing the lens galaxy images above, which uses a grid whose sub-size adaptively
increases depending on a required fractional accuracy of the light profile.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/grid_2d_iterate.py
"""

masked_imaging_iterate = al.MaskedImaging(
    imaging=imaging,
    mask=mask,
    settings=al.SettingsMaskedImaging(grid_class=al.Grid2DIterate),
)

start = time.time()
for i in range(repeats):
    image = lens_galaxy.image_from_grid(grid=masked_imaging_iterate.grid)
    blurring_image = lens_galaxy.image_from_grid(grid=masked_imaging.blurring_grid)

profiling_dict["Lens Light (Grid2DIterate)"] = (time.time() - start) / repeats

"""
__Lens Light Convolution__

Convolves the lens light image above with the PSF and subtracts this from the observed image.

This uses the methods in `Convolver.__init__` and `Convolver.convolve_image`:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/operators/convolver.py
"""
start = time.time()
for i in range(repeats):
    convolved_image = masked_imaging.convolver.convolved_image_from_image_and_blurring_image(
        image=image, blurring_image=blurring_image
    )

profiling_dict["Lens Light Convolution"] = (time.time() - start) / repeats

"""
__Ray Tracing (SIE)__

Compute the deflection angles and ray-trace the image-pixels to the source plane. The run-time of this step depends
on the lens galaxy mass model, for this example we use a fast `EllipticalIsothermal` meaning this step is not the 
bottleneck.

This uses the fast `EllipticalIsothermal` profile.

Deflection angle calculations are profiled fully in the package`profiling/deflections`.

To see examples of deflection angle calculations checkout the `deflections_from_grid` methods at the following link:

https://github.com/Jammy2211/PyAutoGalaxy/blob/master/autogalaxy/profiles/mass_profiles/total_mass_profiles.py

Ray tracing is handled in the following module:

https://github.com/Jammy2211/PyAutoLens/blob/master/autolens/lens/ray_tracing.py

The image-plane pixelization computed below must be ray-traced just like the image-grid and is therefore included in
the profiling time below.
"""
sparse_image_plane_grid = al.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
    grid=masked_imaging.grid, unmasked_sparse_shape=pixelization.shape
)

start = time.time()
for i in range(repeats):
    tracer.deflections_from_grid(grid=sparse_image_plane_grid)
    traced_grid = tracer.traced_grids_of_planes_from_grid(grid=masked_imaging.grid)[-1]

profiling_dict["Ray Tracing (SIE)"] = (time.time() - start) / repeats


"""
__Ray Tracing (Power-Law)__

Compute the deflection angles again, but now using the more expensive `EllipticalPowerLaw` profile.
"""
start = time.time()
for i in range(repeats):
    tracer.deflections_from_grid(grid=sparse_image_plane_grid)
    traced_grid_power_law = tracer_power_law.traced_grids_of_planes_from_grid(
        grid=masked_imaging.grid
    )[-1]

profiling_dict["Ray Tracing (Power-Law)"] = (time.time() - start) / repeats


"""
__Ray Tracing (Decomposed)__

Compute the deflection angles again, now using a very expensive decomposed mass model consisting of 
two `EllipticalSersic`'s and an `EllipticalNFW`.
"""
start = time.time()
for i in range(repeats):
    tracer.deflections_from_grid(grid=sparse_image_plane_grid)
    traced_grid_decomposed = tracer_decomposed.traced_grids_of_planes_from_grid(
        grid=masked_imaging.grid
    )[-1]

profiling_dict["Ray Tracing (Decomposed)"] = (time.time() - start) / repeats

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
        grid=masked_imaging.grid, unmasked_sparse_shape=pixelization.shape
    )

profiling_dict["Image-plane Pixelization (Gridding)"] = (time.time() - start) / repeats

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
__Blurred Mapping Matrix (f_blur)__

For a given source pixel on the mapping matrix, we can use it to map it to a set of image-pixels in the image plane.
This therefore creates a 'image' of the source pixel (which corresponds to a set of values that mostly zeros, but with
1's where mappings occur).

Before reconstructing the source, we blur every one of these source pixel images with the Point Spread Function of our 
dataset via 2D convolution. This uses the methods in `Convolver.__init__` and `Convolver.convolve_mapping_matrix`:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/operators/convolver.py
"""
start = time.time()
for i in range(repeats):
    blurred_mapping_matrix = masked_imaging.convolver.convolve_mapping_matrix(
        mapping_matrix=mapping_matrix
    )
profiling_dict["Blurred Mapping Matrix (f_blur)"] = (time.time() - start) / repeats

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
start = time.time()
subtracted_image = masked_imaging.image - convolved_image
for i in range(repeats):
    data_vector = al.util.inversion.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix,
        image=subtracted_image,
        noise_map=masked_imaging.noise_map,
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
        mapping_matrix=blurred_mapping_matrix, noise_map=masked_imaging.noise_map
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
start = time.time()
for i in range(repeats):
    2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(curvature_reg_matrix))))
profiling_dict["Log Det [F + Lambda H]"] = (time.time() - start) / repeats

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
__Image Reconstruction__

Finally, now we have the reconstructed source pixel fluxes we can map the source flux back to the image plane (via
the blurred mapping_matrix) and reconstruct the image data.

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversions.py
"""
start = time.time()
for i in range(repeats):
    al.util.inversion.mapped_reconstructed_data_from(
        mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
    )
profiling_dict["Image Reconstruction"] = (time.time() - start) / repeats

"""
These two numbers are the primary driver of run time. More pixels = longer run time.
"""

print(f"Inversion fit run times for image type {instrument} \n")
print(f"Number of pixels = {masked_imaging.grid.shape_slim} \n")
print(f"Number of sub-pixels = {masked_imaging.grid.sub_shape_slim} \n")

"""
Print the profiling results of every step of the fit for command line output when running profiling scripts.
"""
for key, value in profiling_dict.items():
    print(key, value)

"""
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
        path=file_path, filename=f"{instrument}_subplot_fit_imaging", format="png"
    )
)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_imaging_plotter.subplot_fit_imaging()

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_of_plane_1", format="png"
    )
)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_imaging_plotter.subplot_of_planes(plane_index=1)

"""
The `info_dict` contains all the key information of the analysis which describes its run times.
"""
info_dict = {}
info_dict["repeats"] = repeats
info_dict["image_pixels"] = masked_imaging.grid.sub_shape_slim
info_dict["sub_size"] = sub_size
info_dict["mask_radius"] = 3.5
info_dict["psf_shape_2d"] = (21, 21)
info_dict["source_pixels"] = len(reconstruction)

print(info_dict)

with open(path.join(file_path, f"{instrument}_info.json"), "w") as outfile:
    json.dump(info_dict, outfile)
