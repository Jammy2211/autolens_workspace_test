"""
Visualization: Interferometer Analysis
=======================================

Tests that `AnalysisInterferometer.visualize_before_fit` and `visualize` output all expected files
to disk and that each output has the correct FITS HDU structure, for a model using an MGE lens bulge +
IsothermalSph mass + rectangular pixelization source on the build interferometer dataset.

A bespoke `config/visualize/plots.yaml` in this directory overrides the repo-level config with
every visualization toggle set to `true`, so all possible outputs are exercised.

Expected outputs are derived directly from the source code of:
  - autolens/interferometer/model/visualizer.py         (VisualizerInterferometer)
  - autolens/interferometer/model/plotter.py  (PlotterInterferometer)
  - autogalaxy/interferometer/plot/fit_interferometer_plots.py (fits_galaxy_images, fits_dirty_images)
  - autolens/analysis/plotter.py              (Plotter: tracer, galaxies, inversion)
  - autogalaxy/analysis/plotter.py            (Plotter: galaxies, inversion)
"""

import os
import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

# Push the bespoke all-true plots.yaml before any visualization method reads config.
from autoconf import conf

conf.instance.push(
    new_path=path.join(path.dirname(path.realpath(__file__)), "config"),
    output_path=path.join(path.dirname(path.realpath(__file__)), "images"),
)

import numpy as np
from astropy.io import fits as astropy_fits

import autofit as af
import autolens as al
from autolens.interferometer.model.visualizer import VisualizerInterferometer


"""
__Dataset__

Build interferometer with_lens_light: data.fits, noise_map.fits, uv_wavelengths.fits, positions.json.
real_space_mask matches the settings used in scripts/interferometer/model_fit.py.
TransformerDFT is used (dataset is small enough for exact DFT).
"""

dataset_path = path.join("dataset", "build", "interferometer", "with_lens_light")

mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(100, 100),
    pixel_scales=0.2,
    radius=mask_radius,
)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)


"""
__Adapt Images__

galaxy_name_image_dict provides per-galaxy images used by adaptive regularization.
dirty_image is the interferometer's real-space image equivalent.
"""

adapt_images = al.AdaptImages(
    galaxy_name_image_dict={
        "('galaxies', 'lens')": dataset.dirty_image,
        "('galaxies', 'source')": dataset.dirty_image,
    },
)


"""
__Positions__

Loaded from positions.json; used to trigger image_with_positions visualization.
"""

positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)
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

analysis = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    adapt_images=adapt_images,
    use_jax=False,
)


"""
__Paths__

Minimal paths stub: VisualizerInterferometer only needs image_path and output_path.
Clean the output directory on each run so assertions reflect this run only.
"""

image_path = Path("scripts") / "interferometer" / "images" / "visualization"

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

Calls PlotterInterferometer.interferometer() -> subplot_dataset.png, dataset.fits
      Plotter.image_with_positions()         -> image_with_positions.png
      Plotter.adapt_images()                 -> subplot_adapt_images.png, adapt_images.fits
"""

VisualizerInterferometer.visualize_before_fit(
    analysis=analysis,
    paths=paths,
    model=model,
)


"""
__Assertions: visualize_before_fit__
"""

# ---- dataset.fits ----
# Source: PlotterInterferometer.interferometer() -> hdu_list_for_output_from with ext_name_list:
#   ["mask", "data", "noise_map", "uv_wavelengths"]
# HDU 0 is PrimaryHDU (mask), HDUs 1-3 are ImageHDU.

assert (image_path / "subplot_dataset.png").exists(), "subplot_dataset.png missing"

with astropy_fits.open(image_path / "dataset.fits") as hdul:
    assert len(hdul) == 4, f"dataset.fits: expected 4 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
    assert hdul[1].name == "DATA"
    assert hdul[2].name == "NOISE_MAP"
    assert hdul[3].name == "UV_WAVELENGTHS"

# ---- image_with_positions.png ----
# Source: Plotter.image_with_positions() -> uses dataset.dirty_image as base

assert (
    image_path / "image_with_positions.png"
).exists(), "image_with_positions.png missing"

# ---- adapt_images.fits ----
# Source: Plotter.adapt_images() -> hdu_list_for_output_from with ext_name_list:
#   ["mask", "('galaxies', 'lens')", "('galaxies', 'source')"]

assert (
    image_path / "subplot_adapt_images.png"
).exists(), "subplot_adapt_images.png missing"

with astropy_fits.open(image_path / "adapt_images.fits") as hdul:
    assert len(hdul) == 3, f"adapt_images.fits: expected 3 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"


"""
__Visualize__

Note: VisualizerInterferometer.visualize() calls fit_interferometer() twice —
once early (with quick_update guard) and once in the full block. The second call
overwrites the first. Both calls execute for during_analysis=False.

Calls PlotterInterferometer.fit_interferometer()
        -> subplot_fit.png              [fit.subplot_fit]
        -> subplot_fit_dirty_images.png [fit_interferometer.subplot_fit_dirty_images]
        -> subplot_fit_real_space.png   [fit_interferometer.subplot_fit_real_space]
        -> subplot_mappings_0.png       [inversion.subplot_mappings]
        -> galaxy_images.fits           [fit.fits_galaxy_images]    (overwritten later by galaxies())
        -> fit_dirty_images.fits        [fit_interferometer.fits_dirty_images]
      Plotter.tracer()
        -> tracer.fits                  [tracer.fits_tracer]
        -> source_plane_images.fits     [tracer.fits_source_plane_images]
        -> subplot_galaxies_images.png  [tracer.subplot_galaxies_images]
      Plotter.galaxies()
        -> subplot_galaxy_images.png    [galaxies.subplot_galaxy_images]
        -> subplot_galaxies.png         [galaxies.subplot_galaxies]
        -> galaxy_images.fits           [galaxies.fits_galaxy_images]  (overwrites fit version)
      Plotter.inversion()
        -> subplot_inversion_0.png      [inversion.subplot_inversion]
        -> source_plane_reconstruction_0.csv  [inversion.csv_reconstruction]
"""

VisualizerInterferometer.visualize(
    analysis=analysis,
    paths=paths,
    instance=instance,
    during_analysis=False,
)


"""
__Assertions: visualize__
"""

# ---- fit_interferometer: PNG subplots ----
# subplot_fit.png               <- FitInterferometer.subplot_fit()            auto_filename="subplot_fit"
# subplot_fit_dirty_images.png  <- FitInterferometer.subplot_fit_dirty_images() auto_filename="subplot_fit_dirty_images"
# subplot_fit_real_space.png    <- FitInterferometer.subplot_fit_real_space() auto_filename="subplot_fit_real_space"
# subplot_mappings_0.png        <- FitInterferometer.subplot_mappings_of_plane() auto_filename="subplot_mappings_{pixelization_index}"

assert (image_path / "subplot_fit.png").exists(), "subplot_fit.png missing"
assert (
    image_path / "subplot_fit_dirty_images.png"
).exists(), "subplot_fit_dirty_images.png missing"
assert (
    image_path / "subplot_fit_real_space.png"
).exists(), "subplot_fit_real_space.png missing"
assert (
    image_path / "subplot_mappings_0.png"
).exists(), "subplot_mappings_0.png missing"

# ---- fit_dirty_images.fits ----
# Source: fits_dirty_images() -> hdu_list_for_output_from with ext_name_list:
#   ["mask", "dirty_image", "dirty_noise_map", "dirty_model_image",
#    "dirty_residual_map", "dirty_normalized_residual_map", "dirty_chi_squared_map"]

with astropy_fits.open(image_path / "fit_dirty_images.fits") as hdul:
    assert len(hdul) == 7, f"fit_dirty_images.fits: expected 7 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
    assert hdul[1].name == "DIRTY_IMAGE"
    assert hdul[2].name == "DIRTY_NOISE_MAP"
    assert hdul[3].name == "DIRTY_MODEL_IMAGE"
    assert hdul[4].name == "DIRTY_RESIDUAL_MAP"
    assert hdul[5].name == "DIRTY_NORMALIZED_RESIDUAL_MAP"
    assert hdul[6].name == "DIRTY_CHI_SQUARED_MAP"
    assert hdul[1].data.ndim == 2, "DIRTY_IMAGE HDU should be 2D"

# ---- tracer.fits ----
# Source: Plotter.tracer() -> hdu_list_for_output_from with ext_name_list:
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
# Source: Plotter.tracer() -> iterates tracer.planes[1:] (source plane only for 2-plane)
# Source galaxy has no LightProfile (pixelization only) so image is zeros.

with astropy_fits.open(image_path / "source_plane_images.fits") as hdul:
    assert len(hdul) == 2, f"source_plane_images.fits: expected 2 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
    assert hdul[1].name == "SOURCE_PLANE_IMAGE_1"

# ---- tracer: subplot_galaxies_images.png ----
# Tracer.subplot_galaxies_images() auto_filename="subplot_galaxies_images"

assert (
    image_path / "subplot_galaxies_images.png"
).exists(), "subplot_galaxies_images.png missing"

# ---- galaxies: PNG subplots ----
# subplot_galaxy_images.png <- Galaxies.subplot_galaxy_images() auto_filename="subplot_galaxy_images"
# subplot_galaxies.png      <- Galaxies.subplot()               auto_filename="subplot_galaxies"

assert (
    image_path / "subplot_galaxy_images.png"
).exists(), "subplot_galaxy_images.png missing"
assert (image_path / "subplot_galaxies.png").exists(), "subplot_galaxies.png missing"

# ---- galaxy_images.fits ----
# Written first by fits_galaxy_images() (fit_interferometer_plots), then overwritten by
# Plotter.galaxies() (galaxies_plots.fits_galaxy_images).
# Final version is from galaxies(): ext_name_list = ["mask", "galaxy_0", "galaxy_1"].

with astropy_fits.open(image_path / "galaxy_images.fits") as hdul:
    assert len(hdul) == 3, f"galaxy_images.fits: expected 3 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
    assert hdul[1].name == "GALAXY_0"
    assert hdul[2].name == "GALAXY_1"

# ---- inversion outputs ----
# subplot_inversion_0.png          <- InversionPlotter.subplot_of_mapper() — plotter appends _0
# source_plane_reconstruction_0.csv <- Plotter.inversion() csv_reconstruction

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
