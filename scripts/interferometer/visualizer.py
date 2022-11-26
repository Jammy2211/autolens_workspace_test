"""
Viusalize: Interferometer
==================

This script performs an interferometer model fit, where all images are output during visualization as .png and .fits
files.

This tests all visualization outputs in **PyAutoLens** for interferometer data.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "visualizer"))

import autofit as af
import autolens as al

"""
__Masking__
"""
real_space_mask_2d = al.Mask2D.circular(
    shape_native=(400, 400), pixel_scales=0.2, radius=3.0, sub_size=1
)

"""
__Dataset__
"""
dataset_label = "build"
dataset_type = "interferometer"
dataset_name = "with_lens_light"

dataset_path = path.join("dataset", dataset_label, dataset_type, dataset_name)

interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask_2d,
    settings=al.SettingsInterferometer(transformer_class=al.TransformerDFT),
)

"""
__Positions__
"""
positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)


"""
__Model__
"""
bulge = af.Model(al.lp.DevVaucouleursSph)

bulge.centre = (0.0, 0.0)
bulge.intensity = 0.1
bulge.effective_radius = 0.8

mass = af.Model(al.mp.IsothermalSph)
mass.centre = (0.0, 0.0)
mass.einstein_radius = 1.6

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.DelaunayMagnification(shape=(30, 30)),
    regularization=al.reg.ConstantSplit,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

instance = model.instance_from_prior_medians()

"""
__Paths__
"""
paths = af.DirectoryPaths(
    path_prefix=path.join("build", "visualizer"), name="interferometer"
)

"""
__Position Likelihood__
"""
positions_likelihood = al.PositionsLHPenalty(positions=positions, threshold=1.0)

"""
__Analysis__
"""
analysis = al.AnalysisInterferometer(
    dataset=interferometer,
    positions_likelihood=positions_likelihood,
    settings_inversion=al.SettingsInversion(
        use_w_tilde=False, use_linear_operators=False
    ),
)

analysis.modify_before_fit(paths=paths, model=model)

analysis.visualize(paths=paths, instance=instance, during_analysis=False)
"""
Finish.
"""
