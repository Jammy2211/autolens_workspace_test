""" 
Find warm pixels in an image from the Hubble Space Telescope (HST) Advanced 
Camera for Surveys (ACS) instrument.

A small patch of the image is plotted with the warm pixels marked with red Xs.
"""

from os import path
import autocti as ac

# Load the HST ACS dataset
dataset_path = path.join("dataset", "examples", "acs", "images")
dataset_name = "jc0a01h8q"
dataset_suffix = "_raw"
frame = ac.acs.ImageACS.from_fits(
    file_path=path.join(dataset_path, dataset_name, f"{dataset_suffix}.fits"),
    quadrant_letter="A",
)

print(frame.shape_native)
extract = frame[:, 50:60]
print(extract.shape)

ac.util.array.numpy_array_2d_to_fits(
    array_2d=extract, file_path=path.join("jacon", "column_x10.fits"), overwrite=True
)
