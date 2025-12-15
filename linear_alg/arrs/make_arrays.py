import numpy as np
from pathlib import Path
import autolens as al

# instrument = "vro"
# instrument = "euclid"
instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

pixel_scales_dict = {"vro": 0.2, "euclid": 0.1, "hst": 0.05, "hst_up": 0.03, "ao": 0.01}
pixel_scale = pixel_scales_dict[instrument]

mesh_shape = (32, 32)

dataset_path = Path("dataset") / "imaging" / "instruments" / instrument

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
    over_sample_size_pixelization=4,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=mask_radius
)

"""
Reshape Dataset so that its exactly paired to the extent PSF convolution goes over including the blurring mask edge.

This speeds up JAX calculations as the PSF convolution is done on a smaller array with fewer zero entries.

This will be put in the source code soon during `apply_mask`.
"""


def false_span(mask: np.ndarray):
    """
    Given a boolean mask with False marking valid pixels,
    return the (y_min, y_max), (x_min, x_max) spans of False entries.
    """
    # Find coordinates of False pixels
    ys, xs = np.where(~mask)

    if ys.size == 0 or xs.size == 0:
        raise ValueError("No False entries in mask!")

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    return (y_max - y_min, x_max - x_min)


y_distance, x_distance = false_span(mask=mask.mask)

(pad_y, pad_x) = dataset.psf.shape_native

new_shape = (y_distance + pad_y, x_distance + pad_x)

mask = mask.resized_from(new_shape=new_shape)
data = dataset.data.resized_from(new_shape=new_shape)
noise_map = dataset.noise_map.resized_from(new_shape=new_shape)

dataset = al.Imaging(
    data=data,
    noise_map=noise_map,
    psf=dataset.psf,
    over_sample_size_pixelization=4,
)

dataset = dataset.apply_mask(mask=mask)

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

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularMagnification(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(
        use_w_tilde=True, use_border_relocator=True
    ),
)

inversion = fit.inversion

mapper = inversion.cls_list_from(al.AbstractMapper)[0]

print(fit.figure_of_merit)

folder = Path("linear_alg") / "arrs" / instrument

np.save(
    f"{folder}/pix_indexes_for_sub_slim_index", mapper.pix_indexes_for_sub_slim_index
)
np.save(f"{folder}/pix_size_for_sub_slim_index", mapper.pix_sizes_for_sub_slim_index)
np.save(
    f"{folder}/pix_weights_for_sub_slim_index", mapper.pix_weights_for_sub_slim_index
)
np.save(f"{folder}/pixels", mapper.pixels)
np.save(
    f"{folder}/total_mask_pixels", mapper.source_plane_data_grid.mask.pixels_in_mask
)
np.save(f"{folder}/slim_index_for_sub_slim_index", mapper.slim_index_for_sub_slim_index)
np.save(f"{folder}/sub_fraction", mapper.over_sampler.sub_fraction.array)
np.save(
    f"{folder}/native_index_for_slim_index", dataset.mask.derive_indexes.native_for_slim
)
np.save(f"{folder}/data_to_pix_unique", mapper.unique_mappings.data_to_pix_unique)
np.save(f"{folder}/data_weights", mapper.unique_mappings.data_weights)
np.save(f"{folder}/pix_lengths", mapper.unique_mappings.pix_lengths)
np.save(f"{folder}/w_matrix", dataset.w_tilde.w_matrix)
np.save(f"{folder}/curvature_preload", dataset.w_tilde.curvature_preload)
np.save(f"{folder}/w_indexes", dataset.w_tilde.indexes)
np.save(f"{folder}/w_lengths", dataset.w_tilde.lengths)
np.save(
    f"{folder}/psf_operator_matrix_dense", dataset.w_tilde.psf_operator_matrix_dense
)
np.save(f"{folder}/mapping_matrix", inversion.mapping_matrix)
np.save(f"{folder}/blurred_mapping_matrix", inversion.operated_mapping_matrix)
np.save(f"{folder}/w_tilde_data", inversion.w_tilde_data)
np.save(f"{folder}/curvature_matrix", inversion.curvature_matrix)
np.save(f"{folder}/regularization_matrix", inversion.regularization_matrix)
np.save(f"{folder}/data_vector", inversion.data_vector)
np.save(f"{folder}/reconstruction", inversion.reconstruction)
np.save(
    f"{folder}/mapped_reconstructed_image", inversion.mapped_reconstructed_image.array
)
np.save(f"{folder}/log_evidence", fit.log_evidence)
