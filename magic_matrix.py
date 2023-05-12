import numpy as np

import autoarray as aa

image=aa.Array2D.ones(shape_native=(7, 7), pixel_scales=(1.0, 1.0))

image[10:40] = 2.0

noise_map=aa.Array2D.full(fill_value=1.0, shape_native=(7, 7), pixel_scales=(1.0, 1.0))

noise_map[23] = 2.0
noise_map[24] = 3.0

psf = np.array([[0.1, 0.3, 0.0], [0.1, 0.3, 0.1], [0.1, 0.1, 0.2]])

psf = aa.Kernel2D.no_mask(values=psf, pixel_scales=(1.0, 1.0))

imaging = aa.Imaging(
        data=image,
    noise_map=noise_map,
        psf=psf )

mask = np.array(
    [
        [True, True, True, True, True, True, True],
        [True, True, False, False, False, True, True],
        [True, True, False, False, False, True, True],
        [True, True, False, False, False, True, True],
        [True, True, False, False, False, True, True],
        [True, True, True, False, True, True, True],
        [True, True, True, True, True, True, True],
    ]
)

mask = aa.Mask2D(mask=mask, sub_size=1, pixel_scales=(1.0, 1.0))

imaging = imaging.apply_mask(mask=mask)
imaging = imaging.apply_settings(settings=aa.SettingsImaging(sub_size_pixelization=2))

print(imaging.noise_map)

# imaging.data[4] = 2.0
# imaging.noise_map[3] = 4.0

mask = imaging.mask

grid = aa.Grid2D.from_mask(mask=mask)

mapping_matrix = np.full(fill_value=0.5, shape=(13, 2))
# mapping_matrix[0, 0] = 0.8

linear_func = aa.m.MockLinearObjFuncList(
    parameters=2, grid=grid, mapping_matrix=mapping_matrix
)

mapper_grids = aa.MapperGrids(
    source_plane_data_grid=imaging.grid,
    source_plane_mesh_grid=aa.Mesh2DRectangular.overlay_grid(
        grid=imaging.grid, shape_native=(3, 3)
    ),
    image_plane_mesh_grid=None,
    hyper_data=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
)

mapper = aa.MapperRectangularNoInterp(
    mapper_grids=mapper_grids, regularization=aa.reg.Constant(coefficient=1.0)
)

inversion_mapping = aa.Inversion(
    dataset=imaging,
    linear_obj_list=[linear_func, mapper],
    settings=aa.SettingsInversion(use_w_tilde=False),
)

inversion_w_tilde = aa.Inversion(
    dataset=imaging,
    linear_obj_list=[linear_func, mapper],
    settings=aa.SettingsInversion(use_w_tilde=True),
)

# For the inversion mapping, the value of curvature matrix[0,1] is:

# 0.4 * 0.3   +    0.45 * 0.1  + 0.45 * 0.1
# 0.125 + 0.045 + 0.045 = 0.21
# 0.21


# print(inversion_mapping.curvature_matrix[1,0])



# For w_tilde we now investigate what goes into this off diagonal term:

operated_mapping_matrix = inversion_w_tilde.linear_func_operated_mapping_matrix_dict[
        linear_func
    ]  / imaging.noise_map[:, None] ** 2

data_to_pix_unique = mapper.unique_mappings.data_to_pix_unique
data_weights = mapper.unique_mappings.data_weights
pix_lengths = mapper.unique_mappings.pix_lengths
pix_pixels = mapper.params
curvature_vector = operated_mapping_matrix

data_pixels = data_weights.shape[0]
linear_func_pixels = curvature_vector.shape[1]

magic_matrix = np.zeros(shape=(data_pixels, linear_func_pixels))

print(operated_mapping_matrix.shape)

for data_0 in range(data_pixels):

    for psf_index in range(inversion_w_tilde.convolver.image_frame_1d_lengths[data_0]):

        data_index = inversion_w_tilde.convolver.image_frame_1d_indexes[data_0, psf_index]
        kernel_value = inversion_w_tilde.convolver.image_frame_1d_kernels[data_0, psf_index]

        for linear_index in range(linear_func_pixels):

            magic_matrix[data_0, linear_index] += kernel_value * operated_mapping_matrix[data_index]

print(magic_matrix[:, 0])

# 0[0.11357659 0.19510191 0.1226167  0.16847469 0.23915187 0.22912558
#  0.10207101 0.13856016 0.1817883 ]


data_to_pix_unique = mapper.unique_mappings.data_to_pix_unique
data_weights = mapper.unique_mappings.data_weights
pix_lengths = mapper.unique_mappings.pix_lengths
pix_pixels = mapper.params

off_diag = np.zeros((pix_pixels, linear_func_pixels))

for data_0 in range(data_pixels):

    for pix_0_index in range(pix_lengths[data_0]):

        data_0_weight = data_weights[data_0, pix_0_index]
        pix_0 = data_to_pix_unique[data_0, pix_0_index]

        for linear_index in range(linear_func_pixels):

            off_diag[pix_0, linear_index] += magic_matrix[data_0, linear_index] * data_0_weight

print()
print(inversion_mapping.curvature_matrix[0, 1:])
print(off_diag[:, 0])

stop

# The preload works as follows.

# If a light profile is fixed, then that means there is a value in an array of shape [total_image_pix, total_linear_light_profiles]
# Which says the following. If the image_pixel value of a mapper pixel is non-zero, then we already have computed the sum of the
# PSF convolutions of that light profiles image-pixels (which via PSF convolution overlap) divided by their noise-map
# values squared.

# Steps:
#
# 1) Implement the for loop aboves in the source code, making sure it works for all case, and profile.
# 2) Implement a function which makes the array described above of shape [total_image_pix, total_linear_light_profiles]
#     and verify it gives values that match the implemented function.
# 3) Write new function which takes as input this preloaded array.
# 4) Implement via preloads.

print(inversion_mapping.curvature_matrix)

print(inversion_mapping.curvature_matrix[0, 1: ] - off_diag)

print()
print("RESULT:")
print(np.sum(np.abs(inversion_mapping.curvature_matrix[1:, 0] - off_diag)))