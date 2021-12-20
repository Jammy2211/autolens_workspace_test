"""
__PGuide: Inversion VoronoiMagnification__

This script provides a step-by-step guide of fitting `Imaging` data with a `VoronoiMagnification` pixelization.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "guide"))

import numpy as np

from autoarray.inversion.mappers.voronoi import MapperVoronoi

import autolens as al
import autolens.plot as aplt

"""
These settings control various aspects of the fit. The values below are default PyAutoLens values.
"""
sub_size = 4
mask_radius = 3.5
psf_shape_2d = (21, 21)
pixelization_shape_2d = (30, 30)

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
The simulated data comes at many resolutions, for this example we'll use euclid resolution.
"""
instrument = "euclid"
pixel_scale = 0.1

"""
Load the dataset for this instrument / resolution.
"""
dataset_path = path.join("dataset", "imaging", "instruments", instrument)

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
    settings=al.SettingsImaging(sub_size=sub_size, sub_size_inversion=sub_size)
)

"""
__Fit__

Perform the complete fit, which this guides break down step-by-step, which we will use to vary that our 
final `log_evidence` values are consistent.

https://github.com/Jammy2211/PyAutoLens/blob/master/autolens/lens/ray_tracing.py
https://github.com/Jammy2211/PyAutoLens/blob/master/autolens/imaging/fit_imaging.py

For this fit, we use the `mapping` formalism which performs the linear algebra via a `mapping_matrix`. The alternative
formalism is called the `w_tilde` formalism, which we turn off below.
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(use_w_tilde=False),
    settings_pixelization=al.SettingsPixelization(use_border=True),
)
fit_log_evidence = fit.log_evidence
print(fit_log_evidence)

"""
__Lens Light (Grid2D)__

Compute a 2D image of the foreground lens galaxy using its light profiles which for this script uses 
an `EllpiticalSersic` bulge and `EllExponential` disk). This computes the `image_2d` of each `LightProfile` and adds 
them together. 

The calculation below uses a `Grid2D` object with a fixed sub-size of 1.

To see examples of `LightProfile` image calculations checkout the `image_2d_from` methods at the following link:

https://github.com/Jammy2211/PyAutoGalaxy/blob/master/autogalaxy/profiles/light_profile_list/light_profile_list.py
"""
image = lens_galaxy.image_2d_from(grid=masked_imaging.grid)

"""
__Lens Light Blurring Grid (Grid2D)__


To convolve the lens's 2D image with a PSF, we also need its `blurring_image` which represents all flux values not 
within the mask, but which are close enough to it that their flux blurs into the mask after PSF convolution. 

To compute this, a `blurring_mask` and `blurring_grid` are used, corresponding to these pixels near the edge of the 
actual mask whose light blurs into the image:

- See the method `blurring_mask_from`: https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/mask/mask_2d.py
- See the method `blurring_grid_from`: https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/grid_2d.py
"""
blurring_image = lens_galaxy.image_2d_from(grid=masked_imaging.blurring_grid)

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

image_iterate = lens_galaxy.image_2d_from(grid=masked_imaging_iterate.grid)
blurring_image_iterate = lens_galaxy.image_2d_from(grid=masked_imaging.blurring_grid)

"""
__Lens Light Convolution__

Convolve the 2D lens light images above with the PSF.

Convolution is performed in real-space (as opposed to via an FFT) using a `Convolver`, which for the data's mask and 
the 2D PSF kernel precomputes all pairings of image pixels that are convolved with one another. 

For the kernel sizes used in lens modeling (e.g. (21,21) or less) real space convolution is faster than an FFT.
Real-space convolution also offers speed up in the linear algebra calculations performed in an inversion, as it can 
exploit sparsity.

This uses the methods in `Convolver.__init__` and `Convolver.convolve_image`. 

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/operators/convolver.py

The convolved image is stored in the fit as the `blurred_image`:

https://github.com/Jammy2211/PyAutoLens/blob/master/autolens/imaging/fit_imaging.py
"""
convolved_image = masked_imaging.convolver.convolve_image(
    image=image, blurring_image=blurring_image
)

"""
__Ray Tracing (SIE)__

Compute the deflection angles and ray-trace the image-pixels to the source plane, using the `EllIsothermal` profile.

To see examples of deflection angle calculations checkout the `deflections_yx_2d_from` methods at the following link:

https://github.com/Jammy2211/PyAutoGalaxy/blob/master/autogalaxy/profiles/mass_profile_list/total_mass_profiles.py

Ray tracing is handled in the following module:

https://github.com/Jammy2211/PyAutoLens/blob/master/autolens/lens/ray_tracing.py

The image-plane pixelization computed below must be ray-traced just like the image-grid and is therefore included in
the profiling time below.
"""
deflections_2d = tracer.deflections_yx_2d_from(grid=masked_imaging.grid)
traced_grid = tracer.traced_grid_list_from(grid=masked_imaging.grid)[-1]

"""
__Ray Tracing Inversion (SIE)__

The grid used to perform an inversion can have a different `sub_size` than the grid used to evaluate light profiles
(e.g. if parametric sources are used in the source plane).

Thus, ray-tracing is performed for a unique grid called `grid_inversion` when performing an `Inversion`.
"""
deflections_2d_inversion = tracer.deflections_yx_2d_from(
    grid=masked_imaging.grid_inversion
)
traced_grid_inversion = tracer.traced_grid_list_from(
    grid=masked_imaging.grid_inversion
)[-1]

"""
__Image-plane Pixelization (Gridding)__

The `VoronoiMagnification` begins by determining what will become its the source-pixel centres by calculating them 
in the image-plane. 

This calculation is performed by overlaying a uniform regular grid with `pixelization_shape_2d` over the image-plane 
mask and retaining all pixels that fall within the mask. This grid is called a `Grid2DSparse` as it retains information
on the mapping between the sparse image-plane pixelization grid and full resolution image grid.

Checkout the functions `Grid2DSparse.__init__` and `Grid2DSparse.from_grid_and_unmasked_2d_grid_shape`

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/grid_2d.py
"""
sparse_image_plane_grid = al.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
    grid=masked_imaging.grid, unmasked_sparse_shape=pixelization.shape
)

"""
__Ray Tracing Sparse Grid (SIE)__

This image-plane pixelization grid is also ray-traced to the source-plane, where its coordinates act as the centres
of the Voronoi cells of the `VoronoiMagnification` pixelization.

The method `traced_sparse_grids_list_from()` returns traced grids of the input sparse grid for every plane.
It returns this as a list of lists of numpy arrays... which is very weird. This needs to be improved, but the reason is
to enable the use of multiple mappers that analysis double source plane lens systems.

For now... this can be ignored.
"""
traced_sparse_grid = tracer.traced_sparse_grid_pg_list_from(
    grid=masked_imaging.grid_inversion
)[0][-1][0]

"""
__Border Relocation__

Coordinates that are ray-traced near the `MassProfile` centre are heavily demagnified and may trace to far outskirts of
the source-plane. We relocate these pixels to the edge of the source-plane border (defined via the border of the 
image-plane mask) have as described in **HowToLens** chapter 4 tutorial 5. 

Checkout the following for a description of the border calculation:

- `border_1d_indexes`: https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/mask/mask_2d.py
- `border_2d_indexes`: https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/mask/mask_2d.py
- `sub_border_flat_indexes`: https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/mask/mask_2d.py
- `sub_border_grid`: https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/abstract_grid_2d.py

Checkout the function `relocated_grid_from` for a full description of the method:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/abstract_grid_2d.py
"""
relocated_grid = traced_grid.relocated_grid_from(grid=traced_grid_inversion)

"""
__Border Relocation Pixelization__

The pixelization grid is also subject to border relocation.

Checkout the function `relocated_pxielization_grid_from` for a full description of the method:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/abstract_grid_2d.py
"""
relocated_pixelization_grid = traced_grid.relocated_pixelization_grid_from(
    pixelization_grid=traced_sparse_grid
)

"""
__Voronoi Mesh__

The relocated pixelization grid is now used to create the `Pixelization`'s Voronoi grid using the scipy.spatial library.

The array `sparse_index_for_slim_index` encodes the closest source pixel of every pixel on the (full resolution)
sub image-plane grid. This is used for efficiently pairing every image-plane pixel to its corresponding source-plane
pixel.

Checkout `Grid2DVoronoi.__init__` and `Grid2DVoronoi.voronoi` property for a full description:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/structures/grids/two_d/grid_2d_pixelization.py
"""
grid_voronoi = al.Grid2DVoronoi(
    grid=relocated_pixelization_grid,
    nearest_pixelization_index_for_slim_index=sparse_image_plane_grid.sparse_index_for_slim_index,
)

"""
__Mapper__

We now combine grids computed above to create a `Mapper`, which describes how every image-plane (sub-)pixel maps to
every source-plane Voronoi pixel. 

There are two steps in this calculation, which we show individually below.

Checkout the modules below for a full description of a `Mapper` and the `mapping_matrix`:

https://github.com/Jammy2211/PyAutoArray/tree/master/autoarray/inversion/mappers
https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/mappers/abstract.py
https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/mappers/voronoi.py
https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/mappers/mapper_util.py
"""
mapper = MapperVoronoi(
    source_grid_slim=relocated_grid,
    source_pixelization_grid=grid_voronoi,
    data_pixelization_grid=sparse_image_plane_grid,  # Only stored in a mapper for visualization of the image-plane grid.
)

"""
__Image-Source Pairing__

The `Mapper` contains:

 1) The traced grid of (y,x) source pixel coordinate centres (`source_grid_slim`).
 2) The traced grid of (y,x) image pixel coordinates (`source_pixelization_grid`).
 
The function below pairs every image-pixel coordinate to every source-pixel centre.

In the API, the `pixelization_index` refers to the source pixel index (e.g. source pixel 0, 1, 2 etc.) whereas the 
sub_slim index refers to the index of a sub-gridded image pixel (e.g. sub pixel 0, 1, 2 etc.). The docstrings of the
function below describes this method.

`MapperVoronoi.pix_index_for_sub_slim_index`: 
https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/mappers/voronoi.py
 
`pixelization_index_for_voronoi_sub_slim_index_from`: 
 https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/util/mapper_util.py 
"""
pix_index_for_sub_slim_index = mapper.pix_index_for_sub_slim_index


"""
__Mapping Matrix (f)__

The `mapping_matrix` is a matrix that represents the image-pixel to source-pixel mappings above in a 2D matrix. 

It has dimensions (total_image_pixels, total_source_pixels).

It is described at the GitHub link below and in the following paper as matrix `f` https://arxiv.org/pdf/astro-ph/0302587.pdf.

`Mapper.__init__`: https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/mappers/abstract.py
`mapping_matrix_from`: https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/mappers/mapper_util.py
"""
mapping_matrix = al.util.mapper.mapping_matrix_from(
    pix_index_for_sub_slim_index=pix_index_for_sub_slim_index,
    pixels=mapper.pixels,
    total_mask_sub_pixels=mapper.source_grid_slim.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=mapper.source_grid_slim.mask.sub_fraction,
)

"""
__Blurred Mapping Matrix (f_blur)__

For a given source pixel on the mapping matrix, we can use it to map it to a set of image-pixels in the image plane.
This therefore creates a 'image' of the source pixel (which corresponds to a set of values that mostly zeros, but with
1's where mappings occur).

Before reconstructing the source, we blur every one of these source pixel images with the Point Spread Function of our 
dataset via 2D convolution. 

This operation does not change the dimensions of the maapping matrix, meaning the `blurred_mapping_matirix` also has
dimensions (total_image_pixels, total_source_pixels).

This uses the methods in `Convolver.__init__` and `Convolver.convolve_mapping_matrix` (it here where our use of real
space convolution can exploit sparcity to speed up the convolution compared to an FFT):

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

The `data_vector` has dimensions (total_source_pixels,).

The calculation is performed by the method `data_vector_via_blurred_mapping_matrix_from` at:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/leq/leq_util.py

This function is called by `LEqMapping.data_vector_from()` to make the `data_vector`:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/leq/imaging.py
"""
subtracted_image = masked_imaging.image - convolved_image
data_vector = al.util.leq.data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix=blurred_mapping_matrix,
    image=subtracted_image,
    noise_map=masked_imaging.noise_map,
)

"""
__Curvature Matrix (F)__

The `curvature_matrix` F is the second matrix, given by equation (4) in https://arxiv.org/pdf/astro-ph/0302587.pdf.

The `curvature_matrix` has dimensions (total_source_pixels, total_source_pixels).

The calculation is performed by the method `curvature_matrix_via_mapping_matrix_from` at:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/leq/leq_util.py

This function is called by `LEqMapping.curvature_matrix` to make the `curvature_matrix`:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/leq/imaging.py
"""
curvature_matrix = al.util.leq.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix, noise_map=masked_imaging.noise_map
)

"""
__Regularization Matrix (H)__

The regularization matrix H is used to impose smoothness on our source reconstruction. This enters the linear algebra
system we solve for using D and F above and is given by equation (12) in https://arxiv.org/pdf/astro-ph/0302587.pdf.

The `regularization_matrix` has dimensions (total_source_pixels, total_source_pixels).

A complete description of regularization is at the link below.

https://github.com/Jammy2211/PyAutoArray/tree/master/autoarray/inversion/regularization
https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/regularization/abstract.py
https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/regularization/constant.py
https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/regularization/regularization_util.py

An `Inversion` object has a property `regularization_matrix` to perform this calculation:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversion/abstract.py
"""
regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
    coefficient=source_galaxy.regularization.coefficient,
    pixel_neighbors=mapper.source_pixelization_grid.pixel_neighbors,
    pixel_neighbors_sizes=mapper.source_pixelization_grid.pixel_neighbors.sizes,
)

"""
__F + Lamdba H__

The linear system of equations solves the `curvature_reg_matrix` F + regularization_coefficient*H.

An `Inversion` object has a property `curvature_reg_matrix` to perform this calculation:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversion/matrices.py
"""
curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

"""
__Source Reconstruction (S)__

Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12) 
of https://arxiv.org/pdf/astro-ph/0302587.pdf 

S is the vector of reconstructed source fluxes and has dimensions (total_source_pixels,).

An `Inversion` object has a property `reconstruction` to perform this calculation:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversion/matrices.py
"""
reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

"""
__Log Det [F + Lambda H]__

The log determinant of [F + reg_coeff*H] is used to determine the Bayesian evidence of the solution.

An `Inversion` object has a property `log_det_curvature_reg_matrix_term` to perform this calculation:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversion/matrices.py
"""
log_curvature_reg_matrix_term = 2.0 * np.sum(
    np.log(np.diag(np.linalg.cholesky(curvature_reg_matrix)))
)

"""
__Log Det [Lambda H]__

The evidence also uses the log determinant of Lambda H.

An `Inversion` object has a property `log_det_regularization_matrix_term` to perform this calculation:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversion/abstract.py
"""
log_regularization_matrix_term = 2.0 * np.sum(
    np.log(np.diag(np.linalg.cholesky(regularization_matrix)))
)

"""
__Regularization Term__

The evidence uses a regularization term, which is the sum of the difference of all reconstructed source pixel fluxes
multiplied by the regularization coefficient.

An `Inversion` object has a property `regularization_term` to perform this calculation:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversion/abstract.py
"""
regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

"""
__Image Reconstruction__

Finally, now we have the reconstructed source pixel fluxes we can map the source flux back to the image plane (via
the blurred mapping_matrix) and reconstruct the image data.

The calculation is performed by the method `mapped_reconstructed_data_via_mapping_matrix_from` at:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/leq/leq_util.py

This function is called by `AbstractInversion.mapped_reconstructed_data`:

https://github.com/Jammy2211/PyAutoArray/blob/master/autoarray/inversion/inversion/abstract.py
"""
mapped_reconstructed_image = al.util.leq.mapped_reconstructed_data_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
)

"""
__Likelihood Calculation__

We now have the model image of our lens light, source reconstruction and the regularization terms to compute our 
final `log_evidence`.

The evidence calculation for a source reconstruction is described in the following papers:

https://arxiv.org/abs/astro-ph/0601493
https://arxiv.org/abs/0804.4002 - equation (5)
"""
model_image = convolved_image + mapped_reconstructed_image

residual_map = masked_imaging.image - model_image
normalized_residual_map = residual_map / masked_imaging.noise_map
chi_squared_map = normalized_residual_map ** 2.0

chi_squared = np.sum(chi_squared_map)

noise_normalization = float(np.sum(np.log(2 * np.pi * masked_imaging.noise_map ** 2.0)))

log_evidence = float(
    -0.5
    * (
        chi_squared
        + regularization_term
        + log_curvature_reg_matrix_term
        - log_regularization_matrix_term
        + noise_normalization
    )
)

print(log_evidence)

"""
__Plots__

We now output images of the fit, so that we can inspect that it fits the data as expected.
"""
plot_path = os.path.join("guide", "inversion", "voronoi_magnification")

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=plot_path, filename=f"{instrument}_subplot_fit_imaging", format="png"
    )
)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_imaging_plotter.subplot_fit_imaging()

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=plot_path, filename=f"{instrument}_subplot_of_plane_1", format="png"
    )
)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_imaging_plotter.subplot_of_planes(plane_index=1)
