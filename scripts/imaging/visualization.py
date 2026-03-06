"""
Visualization: Imaging Analysis
================================

Tests that `AnalysisImaging.visualize_before_fit` and `visualize` output all expected files to disk
and that each output has the correct FITS HDU structure, for a model using an MGE lens bulge +
IsothermalSph mass + rectangular pixelization source on the HST imaging dataset.

A bespoke `config/visualize/plots.yaml` in this directory overrides the repo-level config with
every visualization toggle set to `true`, so all possible outputs are exercised.

Expected outputs are derived directly from the source code of:
  - autolens/imaging/model/visualizer.py    (VisualizerImaging)
  - autolens/imaging/model/plotter_interface.py   (PlotterInterfaceImaging)
  - autolens/analysis/plotter_interface.py        (PlotterInterface: tracer, galaxies, inversion)
  - autogalaxy/analysis/plotter_interface.py      (PlotterInterface: galaxies, inversion)
  - autogalaxy/imaging/model/plotter_interface.py (fits_to_fits)
"""

import os
import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

# Push the bespoke all-true plots.yaml before any visualization method reads config.
# This must come before autolens imports trigger config reads in visualization code paths.
from autoconf import conf

conf.instance.push(
    new_path=path.join(path.dirname(path.realpath(__file__)), "config"),
    output_path=path.join(path.dirname(path.realpath(__file__)), "images"),
)

import numpy as np
from astropy.io import fits as astropy_fits

import autofit as af
import autolens as al
from autolens.imaging.model.visualizer import VisualizerImaging


"""
__Dataset__

HST imaging: pixel_scale=0.05", mask_radius=3.5".
Uses fixed over_sample_size to avoid loading snr_no_lens.fits.
"""

instrument = "hst"
pixel_scale = 0.05

dataset_path = path.join("dataset", "imaging", "instruments", instrument)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scale,
    over_sample_size_lp=2,
    over_sample_size_pixelization=2,
)

mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)


"""
__Adapt Images__

galaxy_name_image_dict provides per-galaxy images used by adaptive regularization.
Keys match the galaxy path strings produced during a real model-fit.
"""

adapt_images = al.AdaptImages(
    galaxy_name_image_dict={
        "('galaxies', 'lens')": dataset.data,
        "('galaxies', 'source')": dataset.data,
    },
)


"""
__Positions__

Two lensed image positions used to trigger image_with_positions visualization.
"""

positions = al.Grid2DIrregular([(-0.5, 1.0), (0.5, -1.0)])
positions_likelihood = al.PositionsLH(positions=positions, threshold=1.0)


"""
__Model__

Lens: MGE bulge (10 Gaussians, 2 per basis) + IsothermalSph mass (fixed centre + einstein_radius).
Source: RectangularAdaptImage 14x14 mesh + Constant regularization.

instance_from_prior_medians() gives a valid instance without running a search.
use_jax=False keeps this test on numpy.
"""

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=10,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

mass = af.Model(al.mp.IsothermalSph)
mass.centre.centre_0 = 0.0
mass.centre.centre_1 = 0.0
mass.einstein_radius = 1.6

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

mesh = al.mesh.RectangularAdaptImage(shape=(14, 14))
regularization = al.reg.Constant(coefficient=1.0)
pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

instance = model.instance_from_prior_medians()


"""
__Analysis__
"""

analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    adapt_images=adapt_images,
    use_jax=False,
)


"""
__Paths__

Minimal paths stub: VisualizerImaging only needs image_path and output_path.
Clean the output directory on each run so assertions reflect this run only.
"""

image_path = Path("scripts") / "imaging" / "images" / "visualization"

if image_path.exists():
    shutil.rmtree(image_path)

image_path.mkdir(parents=True)

output_path = image_path / "output"
output_path.mkdir(parents=True)

paths = SimpleNamespace(
    image_path=image_path,
    output_path=output_path,
)


"""
__Visualize Before Fit__

Calls PlotterInterfaceImaging.imaging()          -> subplot_dataset.png, dataset.fits
      PlotterInterface.image_with_positions()    -> image_with_positions.png
      PlotterInterface.adapt_images()            -> subplot_adapt_images.png, adapt_images.fits
"""

VisualizerImaging.visualize_before_fit(
    analysis=analysis,
    paths=paths,
    model=model,
)


"""
__Assertions: visualize_before_fit__
"""

# ---- dataset.fits ----
# Source: PlotterInterfaceImaging.imaging() -> hdu_list_for_output_from with ext_name_list:
#   ["mask", "data", "noise_map", "psf", "over_sample_size_lp", "over_sample_size_pixelization"]
# HDU 0 is PrimaryHDU (first value), HDUs 1-5 are ImageHDU.

assert (image_path / "subplot_dataset.png").exists(), "subplot_dataset.png missing"

with astropy_fits.open(image_path / "dataset.fits") as hdul:
    assert len(hdul) == 6, f"dataset.fits: expected 6 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
    assert hdul[1].name == "DATA"
    assert hdul[2].name == "NOISE_MAP"
    assert hdul[3].name == "PSF"
    assert hdul[4].name == "OVER_SAMPLE_SIZE_LP"
    assert hdul[5].name == "OVER_SAMPLE_SIZE_PIXELIZATION"
    assert hdul[1].data.ndim == 2, "DATA HDU should be 2D"

# ---- image_with_positions.png ----
# Source: PlotterInterface.image_with_positions() -> image_plotter.set_filename("image_with_positions")

assert (
    image_path / "image_with_positions.png"
).exists(), "image_with_positions.png missing"

# ---- adapt_images.fits ----
# Source: PlotterInterface.adapt_images() -> hdu_list_for_output_from with ext_name_list:
#   ["mask", "('galaxies', 'lens')", "('galaxies', 'source')"]
# HDU 0 = MASK (Primary), HDU 1 = lens key (uppercased), HDU 2 = source key (uppercased).

assert (
    image_path / "subplot_adapt_images.png"
).exists(), "subplot_adapt_images.png missing"

with astropy_fits.open(image_path / "adapt_images.fits") as hdul:
    assert len(hdul) == 3, f"adapt_images.fits: expected 3 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"


"""
__Visualize__

Calls PlotterInterfaceImaging.fit_imaging()  -> subplot_fit.png, subplot_tracer.png,
                                                subplot_fit_log10.png,
                                                subplot_of_plane_0.png, subplot_of_plane_1.png,
                                                subplot_mappings_0.png,
                                                fit.fits, galaxy_images.fits, model_galaxy_images.fits
      PlotterInterface.tracer()              -> tracer.fits, source_plane_images.fits,
                                                subplot_galaxies_images.png
      PlotterInterface.galaxies()            -> subplot_galaxy_images.png, subplot_galaxies.png,
                                                galaxy_images.fits (overwrites fit version)
      PlotterInterface.inversion()           -> subplot_inversion_0.png,
                                                source_plane_reconstruction_0.csv
"""

VisualizerImaging.visualize(
    analysis=analysis,
    paths=paths,
    instance=instance,
    during_analysis=False,
)


"""
__Assertions: visualize__
"""

# ---- fit_imaging: PNG subplots ----
# subplot_fit.png      <- FitImagingPlotter.subplot_fit()       auto_filename="subplot_fit"
# subplot_tracer.png   <- FitImagingPlotter.subplot_tracer()    auto_filename="subplot_tracer"
#                         (called inside fit_imaging when tracer.subplot_tracer=true)
# subplot_fit_log10.png<- FitImagingPlotter.subplot_fit_log10() auto_filename="subplot_fit_log10"
# subplot_of_plane_N.png <- FitImagingPlotter.subplot_of_planes() iterates range(len(tracer.planes))
#                           auto_filename=f"subplot_of_plane_{plane_index}"
# subplot_mappings_0.png <- FitImagingPlotter.subplot_mappings_of_plane()
#                           auto_filename=f"subplot_mappings_{pixelization_index}"

assert (image_path / "subplot_fit.png").exists(), "subplot_fit.png missing"
assert (image_path / "subplot_tracer.png").exists(), "subplot_tracer.png missing"
assert (image_path / "subplot_fit_log10.png").exists(), "subplot_fit_log10.png missing"
assert (
    image_path / "subplot_of_plane_0.png"
).exists(), "subplot_of_plane_0.png missing"
assert (
    image_path / "subplot_of_plane_1.png"
).exists(), "subplot_of_plane_1.png missing"
assert (
    image_path / "subplot_mappings_0.png"
).exists(), "subplot_mappings_0.png missing"

# ---- fit.fits ----
# Source: fits_to_fits() -> hdu_list_for_output_from with ext_name_list:
#   ["mask", "model_data", "residual_map", "normalized_residual_map", "chi_squared_map"]

with astropy_fits.open(image_path / "fit.fits") as hdul:
    assert len(hdul) == 5, f"fit.fits: expected 5 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
    assert hdul[1].name == "MODEL_DATA"
    assert hdul[2].name == "RESIDUAL_MAP"
    assert hdul[3].name == "NORMALIZED_RESIDUAL_MAP"
    assert hdul[4].name == "CHI_SQUARED_MAP"
    assert hdul[1].data.ndim == 2, "MODEL_DATA HDU should be 2D"

# ---- model_galaxy_images.fits ----
# Source: fits_to_fits() -> ext_name_list = ["mask"] + [f"galaxy_{i}" for i in range(number_plots)]
# number_plots = len(galaxy_model_image_dict) + 1, but values_list has len(dict)+1 entries,
# so only len(dict)+1 HDUs are written (loop over values_list). For 2 galaxies: MASK, GALAXY_0, GALAXY_1.

with astropy_fits.open(image_path / "model_galaxy_images.fits") as hdul:
    assert len(hdul) == 3, f"model_galaxy_images.fits: expected 3 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
    assert hdul[1].name == "GALAXY_0"
    assert hdul[2].name == "GALAXY_1"

# ---- tracer.fits ----
# Source: PlotterInterface.tracer() -> hdu_list_for_output_from with ext_name_list:
#   ["mask", "convergence", "potential", "deflections_y", "deflections_x"]

with astropy_fits.open(image_path / "tracer.fits") as hdul:
    assert len(hdul) == 5, f"tracer.fits: expected 5 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
    assert hdul[1].name == "CONVERGENCE"
    assert hdul[2].name == "POTENTIAL"
    assert hdul[3].name == "DEFLECTIONS_Y"
    assert hdul[4].name == "DEFLECTIONS_X"
    assert hdul[1].data.ndim == 2, "CONVERGENCE HDU should be 2D"

# ---- source_plane_images.fits ----
# Source: PlotterInterface.tracer() -> iterates tracer.planes[1:] (source plane only for 2-plane)
# ext_name_list = ["mask", "source_plane_image_1"]
# Source galaxy has no LightProfile (pixelization only) so image is zeros.

with astropy_fits.open(image_path / "source_plane_images.fits") as hdul:
    assert len(hdul) == 2, f"source_plane_images.fits: expected 2 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
    assert hdul[1].name == "SOURCE_PLANE_IMAGE_1"

# ---- tracer: subplot_galaxies_images.png ----
# subplot_galaxies_images.png <- TracerPlotter.subplot_galaxies_images() auto_filename="subplot_galaxies_images"
#                                (triggered by tracer.subplot_galaxies_images=true inside PlotterInterface.tracer())

assert (
    image_path / "subplot_galaxies_images.png"
).exists(), "subplot_galaxies_images.png missing"

# ---- galaxies: PNG subplots ----
# subplot_galaxy_images.png <- GalaxiesPlotter.subplot_galaxy_images() auto_filename="subplot_galaxy_images"
# subplot_galaxies.png      <- GalaxiesPlotter.subplot()               auto_filename="subplot_galaxies"

assert (
    image_path / "subplot_galaxy_images.png"
).exists(), "subplot_galaxy_images.png missing"
assert (image_path / "subplot_galaxies.png").exists(), "subplot_galaxies.png missing"

# ---- galaxy_images.fits ----
# Written first by fits_to_fits() (fit.fits_galaxy_images), then overwritten by PlotterInterface.galaxies()
# (galaxies.fits_galaxy_images). Final version is from galaxies(): ext_name_list = ["mask", "galaxy_0", "galaxy_1"].

with astropy_fits.open(image_path / "galaxy_images.fits") as hdul:
    assert len(hdul) == 3, f"galaxy_images.fits: expected 3 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
    assert hdul[1].name == "GALAXY_0"
    assert hdul[2].name == "GALAXY_1"

# ---- inversion outputs ----
# subplot_inversion_0.png         <- InversionPlotter.subplot_of_mapper(mapper_index=0,
#                                    auto_filename="subplot_inversion") — plotter appends _0
# source_plane_reconstruction_0.csv <- PlotterInterface.inversion() csv_reconstruction

assert (
    image_path / "subplot_inversion_0.png"
).exists(), "subplot_inversion_0.png missing"

assert (
    image_path / "source_plane_reconstruction_0.csv"
).exists(), "source_plane_reconstruction_0.csv missing"

with open(image_path / "source_plane_reconstruction_0.csv") as f:
    header = f.readline().strip()

assert header == "y,x,reconstruction,noise_map", f"Unexpected CSV header: {header}"


print("All visualization assertions passed.")
