""" 
Find and stack warm pixels in bins from a set of images from the Hubble Space 
Telescope (HST) Advanced Camera for Surveys (ACS) instrument.

This is a three-step process:
1. Possible warm pixels are first found in each image (by finding ~delta 
   function local maxima). 
2. The warm pixels are then extracted by ensuring they appear in at least 2/3 of 
   the different images, to discard noise peaks etc. 
3. Finally, they are stacked in bins by (in this example) distance from the 
   readout register and their flux. 

The trails in each bin are then plotted, labelled by their bin start values. 

See the docstrings of find_warm_pixels() in autocti/model/warm_pixels.py and
of find_consistent_lines() and generate_stacked_lines_from_bins() in 
LineCollection in autocti/data/pixel_lines.py for full details.
"""

import numpy as np
import os
from os import path
from autoconf import conf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from urllib.request import urlretrieve

import autocti as ac

# Load the HST ACS dataset
dataset_path = path.join("dataset", "examples", "acs")
dataset_suffix = "_raw"

# Download the example image files
dataset_names = [
    "j9epn8s6q",
    "j9epqbgjq",
    "j9epr7stq",
    # "j9epu6bvq",
    # "j9epn8s7q",
    # "j9epqbgkq",
    # "j9epr7suq",
    # "j9epu6bwq",
    # "j9epn8s9q",
    # "j9epqbgmq",
    # "j9epr7swq",
    # "j9epu6byq",
    # "j9epn8sbq",
    # "j9epqbgoq",
    # "j9epr7syq",
    # "j9epu6c0q",
]

url_path = "http://astro.dur.ac.uk/~cklv53/files/acs"

for dataset_name in dataset_names:
    file = path.join(dataset_path, f"{dataset_name}{dataset_suffix}.fits")
    print(file)
    if not os.path.exists(file):
        print(f"\rDownloading {dataset_name}.fits...", end=" ", flush=True)
        urlretrieve(f"{url_path}/{dataset_name}.fits", file)
print("")

# Initialise the collection of warm pixel trails
warm_pixels = ac.LineCollection()


def prescan_fitted_bias_column(prescan, n_rows=2048, n_rows_ov=20):
    """
    Generate a bias column to be subtracted from the main image by doing a
    least squares fit to the serial prescan region.

    e.g. image -= prescan_fitted_bias_column(image[18:24])

    See Anton & Rorres (2013), S9.3, p460.

    Parameters
    ----------
    prescan : [[float]]
        The serial prescan part of the image. Should usually cover the full
        number of rows but may skip the first few columns of the prescan to
        avoid trails.

    n_rows : int
        The number of rows in the image, exculding overscan.

    n_rows_ov : int, int
        The number of overscan rows in the image.

    Returns
    -------
    bias_column : [float]
        The fitted bias to be subtracted from all image columns.
    """
    n_columns_fit = prescan.shape[1]

    # Flatten the multiple fitting columns to a long 1D array
    # y = [y_1_1, y_2_1, ..., y_nrow_1, y_1_2, y_2_2, ..., y_nrow_ncolfit]
    y = prescan[:-n_rows_ov].T.flatten()
    # x = [1, 2, ..., nrow, 1, ..., nrow, 1, ..., nrow, ...]
    x = np.tile(np.arange(n_rows), n_columns_fit)

    # M = [[1, 1, ..., 1], [x_1, x_2, ..., x_n]].T
    M = np.array([np.ones(n_rows * n_columns_fit), x]).T

    # Best-fit values for y = M v
    v = np.dot(np.linalg.inv(np.dot(M.T, M)), np.dot(M.T, y))

    # Map to full image size for easy subtraction
    bias_column = v[0] + v[1] * np.arange(n_rows + n_rows_ov)

    # print("# fitted bias v =", v)
    # plt.figure()
    # pixels = np.arange(n_rows + n_rows_ov)
    # for i in range(n_columns_fit):
    #     plt.scatter(pixels, prescan[:, i])
    # plt.plot(pixels, bias_column)
    # plt.show()

    return np.transpose([bias_column])


print("1.")
# Find the warm pixels in each image
for dataset_name in dataset_names:
    # Load the HST ACS dataset
    frame = ac.acs.ImageACS.from_fits(
        file_path=f"{dataset_path}/{dataset_name}{dataset_suffix}.fits",
        quadrant_letter="A",
    )
    date = 2400000.5 + frame.exposure_info.modified_julian_date

    # Subtract from all columns the fitted prescan bias
    frame -= prescan_fitted_bias_column(frame[:, 18:24])

    # Load and subtract the bias image
    ###wip

    # Find the warm pixel trails
    new_warm_pixels = ac.find_warm_pixels(image=frame, name=dataset_name, date=date)

    print("Found %d possible warm pixels in %s" % (len(new_warm_pixels), dataset_name))

    # Add them to the collection
    warm_pixels.append(new_warm_pixels)

# For reference, could save the lines and then load at a later time to continue
if not True:
    # Save
    warm_pixels.save("warm_pixel_lines")

    # Load
    warm_pixels = ac.LineCollection()
    warm_pixels.load("warm_pixel_lines")


print("2.")
# Find the consistent warm pixels present in at least 2/3 of the images
consistent_lines = warm_pixels.find_consistent_lines(fraction_present=2 / 3)
print(
    "Found %d consistent warm pixels out of %d possibles"
    % (len(consistent_lines), warm_pixels.n_lines)
)

# Extract the consistent warm pixels
warm_pixels.lines = warm_pixels.lines[consistent_lines]

if not True:
    # Save
    warm_pixels.save("consistent_pixel_lines")

    # Load
    warm_pixels = ac.LineCollection()
    warm_pixels.load("consistent_pixel_lines")


print("3.")
# Stack the lines in bins by distance from readout and total flux
n_row_bins = 5
n_flux_bins = 10
n_background_bins = 2
(
    stacked_lines,
    row_bins,
    flux_bins,
    date_bins,
    background_bins,
) = warm_pixels.generate_stacked_lines_from_bins(
    n_row_bins=n_row_bins,
    n_flux_bins=n_flux_bins,
    n_background_bins=n_background_bins,
    return_bin_info=True,
)

print("Stacked lines in %d bins" % (n_row_bins * n_flux_bins * n_background_bins))

# Plot the stacked trails
plt.figure(figsize=(25, 12))
plt.subplots_adjust(wspace=0, hspace=0)
gs = GridSpec(n_row_bins, n_flux_bins)
axes = [
    [plt.subplot(gs[i_row, i_flux]) for i_flux in range(n_flux_bins)]
    for i_row in range(n_row_bins)
]
length = int(np.amax(stacked_lines.lengths) / 2)
pixels = np.arange(length)
colours = plt.cm.jet(np.linspace(0.05, 0.95, n_background_bins))
y_min = np.amin(stacked_lines.grid_plot)
y_max = 1.5 * np.amax(stacked_lines.grid_plot)

# Plot each stack
for i_row in range(n_row_bins):
    for i_flux in range(n_flux_bins):
        # Furthest row bin at the top
        ax = axes[n_row_bins - 1 - i_row][i_flux]

        for i_background, c in enumerate(colours):
            bin_index = ac.LineCollection.stacked_bin_index(
                i_row=i_row,
                n_row_bins=n_row_bins,
                i_flux=i_flux,
                n_flux_bins=n_flux_bins,
                i_background=i_background,
                n_background_bins=n_background_bins,
            )

            line = stacked_lines.lines[bin_index]

            # Skip empty bins
            if line.n_stacked == 0:
                continue

            ax.errorbar(
                pixels,
                line.grid_plot[length:],
                yerr=line.figure_noise_map[length:],
                c=c,
                capsize=2,
                alpha=0.7,
            )

            # Annotate
            if i_background == 0:
                text = "$N=%d$" % line.n_stacked
            else:
                text = "\n" * i_background + "$%d$" % line.n_stacked
            ax.text(length * 0.9, y_max * 0.7, text, ha="right", va="top")

        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.5, length - 0.5)

        # Axis labels
        if i_flux == 0:
            ax.set_ylabel("Charge")
        else:
            ax.set_yticklabels([])
        if i_row == 0:
            ax.set_xlabel("Pixel")
            ax.set_xticks(np.arange(0, 8.1, 2))
        else:
            ax.set_xticklabels([])

        # Bin labels
        if i_row == n_row_bins - 1:
            ax.xaxis.set_label_position("top")
            ax.set_xlabel(
                "Flux:  %.2g$-$%.2g" % (flux_bins[i_flux], flux_bins[i_flux + 1])
            )
        if i_flux == n_flux_bins - 1:
            ax.yaxis.set_label_position("right")
            text = "Row:  %d$-$%d" % (row_bins[i_row], row_bins[i_row + 1])
            if i_row == int(n_row_bins / 2):
                text += "\n\nBackground:  "
                for i_background in range(n_background_bins):
                    text += "%.0f$-$%.0f" % (
                        background_bins[i_background],
                        background_bins[i_background + 1],
                    )
                    if i_background < n_background_bins - 1:
                        text += ",  "
            ax.set_ylabel(text)

"""Output image of stack warm pixel lines"""

output_path = path.join("dataset", "examples", "acs", "lines")

plt.savefig(path.join(output_path, "stack_warm_pixels.png"), dpi=400)
plt.close()
print("Saved" + path.join(output_path, "stack_warm_pixels.png"))

"""Serialize the `LineCollection` so we can load it in the CTI model fitting scripts."""

if not os.path.exists(output_path):
    os.mkdir(output_path)

warm_pixels.save(filename=path.join(output_path, f"{dataset_name}_stacked.pickle"))
