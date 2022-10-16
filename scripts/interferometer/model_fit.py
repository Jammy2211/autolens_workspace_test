"""
Modeling: Mass Total + Source Inversion
=======================================

This script fits `Interferometer` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `DelaunayMagnification` `Pixelization` and `Constant`
   regularization.
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

conf.instance.push(new_path=path.join(cwd, "config", "searches"))

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Masking__

We define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
real_space_mask_2d = al.Mask2D.circular(
    shape_native=(100, 100), pixel_scales=0.2, radius=3.0, sub_size=1
)

"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `mass_sie__source_sersic` from .fits files , which we will fit 
with the lens model.
"""
dataset_label = "build"
dataset_type = "interferometer"
dataset_name = "no_lens_light"

dataset_path = path.join("dataset", dataset_label, dataset_type, dataset_name)

interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask_2d,
)

"""
__Inversion Settings (Run Times)__

"""
settings_interferometer = al.SettingsInterferometer(transformer_class=al.TransformerDFT)
settings_inversion = al.SettingsInversion(use_linear_operators=False)

"""
We now create the `Interferometer` object which is used to fit the lens model.

This includes a `SettingsInterferometer`, which includes the method used to Fourier transform the real-space 
image of the strong lens to the uv-plane and compare directly to the visiblities. We use a non-uniform fast Fourier 
transform, which is the most efficient method for interferometer datasets containing ~1-10 million visibilities.
"""
interferometer = interferometer.apply_settings(settings=settings_interferometer)

"""
__Positions__

This fit also uses the arc-second positions of the multiply imaged lensed source galaxy, which were drawn onto the
image via the GUI described in the file `autolens_workspace/*/imaging/preprocess/gui/positions.py`.
"""
positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)

"""
__Model__

"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=al.mp.SphIsothermal,
)

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.DelaunayMagnification(shape=(30, 30)),
    regularization=al.reg.ConstantSplit,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search__

The model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Dynesty (https://dynesty.readthedocs.io/en/latest/).

A full description of the settings below is given in the beginner modeling scripts, if anything is unclear.
"""
search = af.DynestyStatic(
    path_prefix=path.join("build", "model_fit", "interferometer"),
    name=dataset_name,
    nlive=50,
    number_of_cores=2,
)

"""
__Position Likelihood__

"""
positions_likelihood = al.PositionsLHPenalty(positions=positions, threshold=0.1)

"""
__Analysis__

The `AnalysisInterferometer` object defines the `log_likelihood_function` used by the non-linear search to fit the 
model to the `Interferometer`dataset.
"""
analysis = al.AnalysisInterferometer(
    dataset=interferometer,
    positions_likelihood=positions_likelihood,
    settings_inversion=settings_inversion,
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the output folder for live outputs of the results of the fit, including on-the-fly visualization of the best 
fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=real_space_mask_2d.masked_grid_sub_1
)
tracer_plotter.subplot_tracer()

fit_interferometer_plotter = aplt.FitInterferometerPlotter(
    fit=result.max_log_likelihood_fit
)
fit_interferometer_plotter.subplot_fit_interferometer()
fit_interferometer_plotter.subplot_fit_dirty_images()

dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
dynesty_plotter.cornerplot()

os._exit(1)

"""
Checkout `autolens_workspace/*/results` for a full description of analysing results in **PyAutoLens**.
"""
