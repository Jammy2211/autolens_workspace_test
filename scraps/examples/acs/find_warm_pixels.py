""" 
Find warm pixels in an image from the Hubble Space Telescope (HST) Advanced 
Camera for Surveys (ACS) instrument.

A small patch of the image is plotted with the warm pixels marked with red Xs.
"""

import numpy as np
from os import path
import os
import matplotlib.pyplot as plt

import autocti as ac

# Load the HST ACS dataset
dataset_path = path.join("dataset", "examples", "acs", "images")
dataset_name = "jc0a01h8q_raw"

frame = ac.acs.ImageACS.from_fits(
    file_path=path.join(dataset_path, dataset_name, ".fits"), quadrant_letter="A"
)

bias_frame = ac.acs.FrameACS.from_fits(
    file_path=path.join(dataset_path, "2581509lj_bia.fits"), quadrant_letter="A"
)

frame = frame - bias_frame


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

    print("# fitted bias v =", v)
    # plt.figure()
    # pixels = np.arange(n_rows + n_rows_ov)
    # for i in range(n_columns_fit):
    #     plt.scatter(pixels, prescan[:, i])
    # plt.plot(pixels, bias_column)
    # plt.show()

    return np.transpose([bias_column])


# Subtract from all columns the fitted prescan bias
frame -= prescan_fitted_bias_column(frame[:, 18:24])

# Load and subtract the bias image
###wip
# bias_name = "acs/25b1734qj_bia"
# bias = acs.ImageACS.from_fits(file_path=f"{bias_name}.fits", quadrant_letter="A")
# print(np.amin(bias), np.mean(bias), np.median(bias), np.amax(bias))
# frame -= bias

# Extract an example patch of the full image
row_start, row_end, column_start, column_end = -300, -100, -300, -100
frame = frame[row_start:row_end, column_start:column_end]
frame.mask = frame.mask[row_start:row_end, column_start:column_end]

# Find the warm pixel trails and store in a line collection object
warm_pixels = ac.LineCollection(lines=ac.find_warm_pixels(image=frame))

print("Found %d warm pixels" % warm_pixels.n_lines)

"""Output image of stack war pixel lines"""

output_path = "dataset/examples/acs/lines"

# Plot the image and the found warm pixels
plt.figure()
im = plt.imshow(X=frame, aspect="equal", vmin=0, vmax=500)
plt.scatter(
    warm_pixels.locations[:, 1],
    warm_pixels.locations[:, 0],
    c="r",
    marker="x",
    s=4,
    linewidth=0.2,
)
plt.colorbar(im)
plt.axis("off")
plt.savefig(path.join(output_path, "find_warm_pixels.png"), dpi=400)
plt.close()
print("Saved" + path.join(output_path, "find_warm_pixels.png"))

"""
Serialize the `LineCollection` so we can load it and convert it to a `ImagingCI` dataset in the script 
`line_collection_to_ci.py`.
"""

if not os.path.exists(output_path):
    os.mkdir(output_path)

warm_pixels.save(filename=path.join(output_path, dataset_name, ".pickle"))
