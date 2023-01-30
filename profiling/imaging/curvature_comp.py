"""
__PROFILING: Inversion DelaunayMagnification__

This profiling script times how long it takes to fit `Imaging` data with a `DelaunayMagnification` mesh for
datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoLens** log likelihood function.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import time
import json

import autolens as al
import autolens.plot as aplt


"""
The path all profiling results are output.
"""
profiling_path = path.dirname(path.realpath(__file__))

file_path = os.path.join(profiling_path, "times", al.__version__)

"""
Whether w_tilde is used dictates the output folder.
"""
use_w_tilde = True

"""
Whether the lens light is a linear object or not.
"""

bulge_cls = al.lp_linear.Sersic
# disk_cls = al.lp.Exponential

file_path = os.path.join(file_path, "lens_light_linear")

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 4
mask_radius = 2.0
psf_shape_2d = (21, 21)
mesh_shape_2d = (40, 40)


print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"psf shape = {psf_shape_2d}")
print(f"pixelization shape = {mesh_shape_2d}")

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=bulge_cls(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    # disk=disk_cls(
    #     centre=(0.0, 0.0),
    #     ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
    #     effective_radius=1.6,
    # ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.001, gamma_2=0.001),
)

"""
The source galaxy whose `DelaunayMagnification` `Pixelization` fits the data.
"""
mesh = al.mesh.DelaunayMagnification(shape=mesh_shape_2d)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.Pixelization(
        mesh=mesh, regularization=al.reg.ConstantSplit(coefficient=1.0)
    ),
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
    settings=al.SettingsImaging(sub_size=sub_size)
)

"""
__Numba Caching__

Call FitImaging once to get all numba functions initialized.
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit_mapping = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(use_w_tilde=False),
)

fit_w_tilde = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(use_w_tilde=True),
)


import numpy as np

# print(fit_mapping.inversion.data_vector - fit_w_tilde.inversion.data_vector)
# print(np.max(np.abs(fit_mapping.inversion.data_vector - fit_w_tilde.inversion.data_vector)))


import autoarray as aa

mapper_index_range = fit_w_tilde.inversion.index_range_list_from(cls=al.AbstractMapper)[0]
func_index_range = fit_w_tilde.inversion.index_range_list_from(cls=aa.AbstractLinearObjFuncList)[0]


print("lp_linear diag:")
print(fit_mapping.inversion.curvature_matrix[0:1,0:1] - fit_w_tilde.inversion.curvature_matrix[0:1,0:1])
print(np.max(np.abs(fit_mapping.inversion.curvature_matrix[0:1,0:1] - fit_w_tilde.inversion.curvature_matrix[0:1,0:1])))

print("pix diag:")
print(fit_mapping.inversion.curvature_matrix[1:-1,1:-1] - fit_w_tilde.inversion.curvature_matrix[1:-1,1:-1])
print(np.max(np.abs(fit_mapping.inversion.curvature_matrix[1:-1,1:-1] - fit_w_tilde.inversion.curvature_matrix[1:-1,1:-1])))

print("lp_linear mapper off-diag:")
print(fit_mapping.inversion.curvature_matrix[0:1,1:-1] - fit_w_tilde.inversion.curvature_matrix[0:1,1:-1])
print(np.max(np.abs(fit_mapping.inversion.curvature_matrix[0:1,1:-1] - fit_w_tilde.inversion.curvature_matrix[0:1,1:-1])))

print("lp_linear mapper off-diag (transpoe):")
print(fit_mapping.inversion.curvature_matrix[1:-1, 0:1] - fit_w_tilde.inversion.curvature_matrix[1:-1, 0:1])
print(np.max(np.abs(fit_mapping.inversion.curvature_matrix[1:-1, 0:1] - fit_w_tilde.inversion.curvature_matrix[1:-1, 0:1])))

print("Whole Matrix:")
print(fit_mapping.inversion.curvature_matrix - fit_w_tilde.inversion.curvature_matrix)
print(np.max(np.abs(fit_mapping.inversion.curvature_matrix - fit_w_tilde.inversion.curvature_matrix)))

print(fit_mapping.figure_of_merit)
print(fit_w_tilde.figure_of_merit)