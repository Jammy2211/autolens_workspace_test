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
import autolens.plot as aplt
import numpy as np

"""
__Masking__
"""
real_space_mask_2d = al.Mask2D.circular(
    shape_native=(400, 400), pixel_scales=0.2, radius=3.0, sub_size=1
)

"""
__Dataset__
"""
dataset_name = "light_sersic_exp__mass_sie__source_sersic"
dataset_path = path.join("dataset", "interferometer", dataset_name)

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
bulge = af.Model(al.lp.EllSersic)

bulge.centre = (0.0, 0.0)
bulge.elliptical_comps = al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0)
bulge.intensity = 4.0
bulge.effective_radius = 0.6
bulge.sersic_index = 3.0

disk = af.Model(al.lp_linear.EllExponential)
disk.centre = (0.05, 0.05)
disk.elliptical_comps = al.convert.elliptical_comps_from(axis_ratio=0.7, angle=30.0)
disk.effective_radius = 1.6

mass = af.Model(al.mp.EllIsothermal)
mass.centre = (0.0, 0.0)
mass.einstein_radius = 1.6
mass.elliptical_comps = al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0)

shear = af.Model(al.mp.ExternalShear)
shear.elliptical_comps = (0.05, 0.05)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, disk=disk, mass=mass, shear=shear)
pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.VoronoiNNMagnification(shape=(30, 30)),
    regularization=al.reg.ConstantSplit,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search__
"""
search = af.DynestyStatic(
    path_prefix=path.join("visualizer"),
    name="interferometer",
    unique_tag=dataset_name,
    nlive=50,
    number_of_cores=1,
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

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
