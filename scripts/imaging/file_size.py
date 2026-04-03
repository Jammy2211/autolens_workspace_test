"""
File Size Investigation: subplot_fit and subplot_fit_log10
==========================================================

Generates subplot_fit.png and subplot_fit_log10.png, reports their file sizes,
then investigates how DPI reduction and PNG compression level affect size while
preserving visual quality.

Run from the repository root:

    python scripts/imaging/file_size.py
"""

import io
import os
import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

from autoconf import conf

conf.instance.push(
    new_path=path.join(path.dirname(path.realpath(__file__)), "config"),
    output_path=path.join(path.dirname(path.realpath(__file__)), "images"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

import autofit as af
import autolens as al
from autolens.imaging.plot.fit_imaging_plots import (
    subplot_fit,
    subplot_fit_log10,
    _compute_critical_curve_lines,
)
from autoarray.plot.utils import conf_subplot_figsize, save_figure


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.5,
)
dataset = dataset.apply_mask(mask=mask)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

bulge = al.model_util.mge_model_from(
    mask_radius=3.5,
    total_gaussians=10,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

mass = af.Model(al.mp.PowerLaw)
mass.centre.centre_0 = 0.0
mass.centre.centre_1 = 0.0
mass.ell_comps.ell_comps_0 = 0.05
mass.ell_comps.ell_comps_1 = 0.05
mass.einstein_radius = 1.6
mass.slope = 1.8

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

image_mesh = al.image_mesh.Overlay(shape=(26, 26))
image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=dataset.mask)

mesh = al.mesh.Delaunay(pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=0)
regularization = al.reg.ConstantSplit(coefficient=1.0)
pixelization = al.Pixelization(mesh=mesh, regularization=regularization)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
instance = model.instance_from_prior_medians()

adapt_images = al.AdaptImages(
    galaxy_name_image_dict={
        "('galaxies', 'lens')": dataset.data,
        "('galaxies', 'source')": dataset.data,
    },
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    use_jax=True,
)

fit = analysis.fit_from(instance=instance)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

out_dir = Path("scripts") / "imaging" / "images" / "file_size"
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_kb(n_bytes: int) -> str:
    return f"{n_bytes / 1024:.1f} KB"


def compress_png(src: Path, dst: Path, compress_level: int, optimize: bool = True) -> int:
    """Re-save a PNG with a given PIL compress_level (0-9) and return new byte size."""
    img = Image.open(src)
    img.save(dst, format="PNG", compress_level=compress_level, optimize=optimize)
    return dst.stat().st_size


def savefig_dpi(fig, dst: Path, dpi: int) -> int:
    """Save a matplotlib figure at a given DPI and return byte size."""
    fig.savefig(dst, dpi=dpi, bbox_inches="tight")
    return dst.stat().st_size


# ---------------------------------------------------------------------------
# Build the figure objects directly so we can re-save at multiple DPIs
# without recomputing the fit.  Mirrors what subplot_fit / subplot_fit_log10
# do internally, but retains the figure handle.
# ---------------------------------------------------------------------------

import autoarray as aa
from autolens.imaging.plot.fit_imaging_plots import _get_source_vmax, _plot_source_plane, _symmetric_vmax
from autoarray.plot.array import plot_array
from autoarray.plot.utils import hide_unused_axes

_zoom = aa.Zoom2D(mask=fit.mask)
_cc_grid = aa.Grid2D.from_extent(
    extent=_zoom.extent_from(buffer=0), shape_native=_zoom.shape_native
)
tracer = fit.tracer_linear_light_profiles_to_light_profiles
ip_lines, ip_colors, sp_lines, sp_colors = _compute_critical_curve_lines(tracer, _cc_grid)

source_vmax = _get_source_vmax(fit)
final_plane_index = len(fit.tracer.planes) - 1

print("\n" + "=" * 60)
print("  subplot_fit.png  —  file size investigation")
print("=" * 60)


def build_fit_fig():
    fig, axes = plt.subplots(3, 4, figsize=conf_subplot_figsize(3, 4))
    ax = list(axes.flatten())
    plot_array(array=fit.data, ax=ax[0], title="Data")
    plot_array(array=fit.data, ax=ax[1], title="Data (Source Scale)", vmax=source_vmax)
    plot_array(array=fit.signal_to_noise_map, ax=ax[2], title="Signal-To-Noise Map")
    plot_array(array=fit.model_data, ax=ax[3], title="Model Image",
               lines=ip_lines, line_colors=ip_colors)
    try:
        plot_array(array=fit.model_images_of_planes_list[0], ax=ax[4],
                   title="Lens Light Model Image")
    except (IndexError, AttributeError):
        ax[4].axis("off")
    try:
        sub = fit.subtracted_images_of_planes_list[final_plane_index]
        plot_array(array=sub, ax=ax[5], title="Lens Light Subtracted",
                   vmin=0.0 if source_vmax is not None else None, vmax=source_vmax)
    except (IndexError, AttributeError):
        ax[5].axis("off")
    try:
        plot_array(array=fit.model_images_of_planes_list[final_plane_index], ax=ax[6],
                   title="Source Model Image", vmax=source_vmax,
                   lines=ip_lines, line_colors=ip_colors)
    except (IndexError, AttributeError):
        ax[6].axis("off")
    _plot_source_plane(fit, ax[7], final_plane_index, zoom_to_brightest=True,
                       title="Source Plane (Zoomed)",
                       lines=sp_lines, line_colors=sp_colors)
    norm_resid = fit.normalized_residual_map
    _abs_max = _symmetric_vmax(norm_resid)
    plot_array(array=norm_resid, ax=ax[8], title="Normalized Residual Map",
               vmin=-_abs_max, vmax=_abs_max)
    plot_array(array=norm_resid, ax=ax[9], title=r"Normalized Residual Map $1\sigma$",
               vmin=-1.0, vmax=1.0)
    plot_array(array=fit.chi_squared_map, ax=ax[10], title="Chi-Squared Map",
               cb_unit=r"$\chi^2$")
    _plot_source_plane(fit, ax[11], final_plane_index, zoom_to_brightest=False,
                       title="Source Plane (No Zoom)",
                       lines=sp_lines, line_colors=sp_colors)
    hide_unused_axes(ax)
    plt.tight_layout()
    return fig


def build_fit_log10_fig():
    fig, axes = plt.subplots(3, 4, figsize=conf_subplot_figsize(3, 4))
    ax = list(axes.flatten())
    plot_array(array=fit.data, ax=ax[0], title="Data", use_log10=True)
    try:
        plot_array(array=fit.data, ax=ax[1], title="Data (Source Scale)", use_log10=True)
    except ValueError:
        ax[1].axis("off")
    try:
        plot_array(array=fit.signal_to_noise_map, ax=ax[2], title="Signal-To-Noise Map",
                   use_log10=True)
    except ValueError:
        ax[2].axis("off")
    plot_array(array=fit.model_data, ax=ax[3], title="Model Image", use_log10=True,
               lines=ip_lines, line_colors=ip_colors)
    try:
        plot_array(array=fit.model_images_of_planes_list[0], ax=ax[4],
                   title="Lens Light Model Image", use_log10=True)
    except (IndexError, AttributeError):
        ax[4].axis("off")
    try:
        sub = fit.subtracted_images_of_planes_list[final_plane_index]
        plot_array(array=sub, ax=ax[5], title="Lens Light Subtracted", use_log10=True)
    except (IndexError, AttributeError):
        ax[5].axis("off")
    try:
        plot_array(array=fit.model_images_of_planes_list[final_plane_index], ax=ax[6],
                   title="Source Model Image", use_log10=True,
                   lines=ip_lines, line_colors=ip_colors)
    except (IndexError, AttributeError):
        ax[6].axis("off")
    _plot_source_plane(fit, ax[7], final_plane_index, zoom_to_brightest=True,
                       use_log10=True, lines=sp_lines, line_colors=sp_colors,
                       title="Source Plane (Zoomed)")
    norm_resid = fit.normalized_residual_map
    _abs_max = _symmetric_vmax(norm_resid)
    plot_array(array=norm_resid, ax=ax[8], title="Normalized Residual Map",
               vmin=-_abs_max, vmax=_abs_max, cb_unit=r"$\sigma$")
    plot_array(array=norm_resid, ax=ax[9], title=r"Normalized Residual Map $1\sigma$",
               vmin=-1.0, vmax=1.0, cb_unit=r"$\sigma$")
    plot_array(array=fit.chi_squared_map, ax=ax[10], title="Chi-Squared Map",
               use_log10=True, cb_unit=r"$\chi^2$")
    _plot_source_plane(fit, ax[11], final_plane_index, zoom_to_brightest=False,
                       use_log10=True, lines=sp_lines, line_colors=sp_colors,
                       title="Source Plane (No Zoom)")
    hide_unused_axes(ax)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Run investigation for both figures
# ---------------------------------------------------------------------------

DPIS = [300, 200, 150, 100]
COMPRESS_LEVELS = [6, 7, 8, 9]  # PNG compress_level (higher = smaller but slower)

for fig_name, build_fn in [("subplot_fit", build_fit_fig),
                             ("subplot_fit_log10", build_fit_log10_fig)]:

    print(f"\n{'─'*50}")
    print(f"  {fig_name}.png")
    print(f"{'─'*50}")

    # --- Baseline at DPI 300 ---
    baseline_path = out_dir / f"{fig_name}_dpi300.png"
    fig = build_fn()
    size_300 = savefig_dpi(fig, baseline_path, dpi=300)
    plt.close("all")
    print(f"\nBaseline DPI=300:  {fmt_kb(size_300)}")

    # --- DPI variants ---
    print("\nDPI reduction:")
    for dpi in DPIS[1:]:
        dst = out_dir / f"{fig_name}_dpi{dpi}.png"
        fig = build_fn()
        sz = savefig_dpi(fig, dst, dpi=dpi)
        plt.close("all")
        pct = 100 * sz / size_300
        print(f"  DPI={dpi:3d}: {fmt_kb(sz):>10}  ({pct:.0f}% of baseline)")

    # --- PNG compress_level variants (lossless, DPI=300) ---
    print("\nPNG compress_level (DPI=300, lossless):")
    for level in COMPRESS_LEVELS:
        dst = out_dir / f"{fig_name}_compress{level}.png"
        sz = compress_png(baseline_path, dst, compress_level=level)
        pct = 100 * sz / size_300
        print(f"  compress_level={level}: {fmt_kb(sz):>10}  ({pct:.0f}% of baseline)")

    # --- Best combination: lower DPI + high compress ---
    print("\nBest combination (DPI=150, compress_level=9):")
    combined_path = out_dir / f"{fig_name}_dpi150_compress9.png"
    fig = build_fn()
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    img = Image.open(combined_path)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=9, optimize=True)
    combined_path.write_bytes(buf.getvalue())
    sz_combined = combined_path.stat().st_size
    pct = 100 * sz_combined / size_300
    print(f"  {fmt_kb(sz_combined):>10}  ({pct:.0f}% of baseline)")


# ---------------------------------------------------------------------------
# Visual comparison: save a side-by-side comparison PNG and open it
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  Visual comparison — saving comparison image")
print("=" * 60)

fig_compare, axes = plt.subplots(2, 2, figsize=(20, 16))
fig_compare.suptitle(
    "File size investigation — visual comparison\n"
    "Left: baseline DPI=300  |  Right: DPI=150 + compress_level=9",
    fontsize=13,
)

for row, fig_name in enumerate(["subplot_fit", "subplot_fit_log10"]):
    baseline = mpimg.imread(str(out_dir / f"{fig_name}_dpi300.png"))
    compressed = mpimg.imread(str(out_dir / f"{fig_name}_dpi150_compress9.png"))

    sz_orig = (out_dir / f"{fig_name}_dpi300.png").stat().st_size
    sz_comp = (out_dir / f"{fig_name}_dpi150_compress9.png").stat().st_size

    axes[row, 0].imshow(baseline)
    axes[row, 0].set_title(
        f"{fig_name} — DPI 300\n{fmt_kb(sz_orig)}", fontsize=10
    )
    axes[row, 0].axis("off")

    axes[row, 1].imshow(compressed)
    axes[row, 1].set_title(
        f"{fig_name} — DPI 150 + compress 9\n{fmt_kb(sz_comp)}", fontsize=10
    )
    axes[row, 1].axis("off")

plt.tight_layout()
comparison_path = out_dir / "comparison.png"
fig_compare.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"\nComparison image saved to: {comparison_path}")

# Open with the OS default viewer (Windows explorer in WSL, xdg-open on Linux).
import subprocess
import sys

if sys.platform.startswith("linux"):
    try:
        # WSL: convert to Windows path and open with explorer.exe
        win_path = subprocess.check_output(
            ["wslpath", "-w", str(comparison_path.resolve())],
            text=True,
        ).strip()
        subprocess.Popen(["explorer.exe", win_path])
    except Exception:
        try:
            subprocess.Popen(["xdg-open", str(comparison_path.resolve())])
        except Exception:
            print("Could not open image automatically — open manually:", comparison_path)
elif sys.platform == "darwin":
    subprocess.Popen(["open", str(comparison_path.resolve())])
else:
    subprocess.Popen(["start", str(comparison_path.resolve())], shell=True)
