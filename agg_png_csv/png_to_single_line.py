import json
import os
from os import path
from PIL import Image

import png_util

dataset_core_path = path.join("agg_png_csv", "dataset", "lens")

for dataset_name in os.listdir(dataset_core_path):

    print()
    print(dataset_name)

    dataset_fits_name = f"{dataset_name}.fits"

    dataset_main_path = path.join(dataset_core_path, dataset_name)
    dataset_result_path = path.join(dataset_main_path, "result")

    """
    __MGE Fit__
    """
    image_top = Image.open(path.join(dataset_result_path, "sie_fit_mge.png"))

    new_order = [1, 2, 3]

    original_shape = (4, 6)  # Number of rows and columns
    new_shape = (1, 3)

    image_top = png_util.make_png(image_top, original_shape, new_order, new_shape)

    """
    __RGB 0__
    """
    additional_file = path.join(dataset_main_path, "rgb_0.png")
    additional_img = Image.open(additional_file)

    zoom_factor = 2.5
    additional_img = png_util.zoom_image(additional_img, zoom_factor)

    image_top = png_util.add_image_to_left(image_top, additional_img)

    """
    __Pix Fit__
    """
    image_bottom = Image.open(path.join(dataset_result_path, "sie_fit_pix.png"))


    new_order = [1, 2, 3]

    original_shape = (4, 6)  # Number of rows and columns
    new_shape = (1, 3)

    image_bottom = png_util.make_png(image_bottom, original_shape, new_order, new_shape)

    """
    __RGB 1__
    """
    additional_file = path.join(dataset_main_path, "rgb_1.png")
    additional_img = Image.open(additional_file)

    zoom_factor = 2.5
    additional_img = png_util.zoom_image(additional_img, zoom_factor)

    image_bottom = png_util.add_image_to_left(image_bottom, additional_img)

    """
    __STack__
    """
    image = png_util.stack_images(images=[image_top, image_bottom])

    # Save the new image
    output_file = path.join("agg_png_csv", "png", f"{dataset_name}.png")
    image.save(output_file)
    print(f"Rearranged image saved as {output_file}")

