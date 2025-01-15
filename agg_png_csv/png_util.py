import os
from PIL import Image

def dataset_list_from():
    dataset_list = list(reversed(os.listdir("dataset")))

    # if begins with M24 move to front of list

    dataset_list = sorted(dataset_list, key=lambda x: not x.startswith("M24_"))

    return dataset_list

def make_png(image, original_shape, new_order, new_shape):

    rows, cols = original_shape
    new_rows, new_cols = new_shape

    width, height = image.size
    subplot_width = width // cols
    subplot_height = height // rows

    def extract_subplot(image, row, col):
        left = col * subplot_width
        upper = row * subplot_height
        right = left + subplot_width
        lower = upper + subplot_height
        return image.crop((left, upper, right, lower))

    # Extract all subplots
    subplots = []
    for r in range(rows):
        for c in range(cols):
            subplots.append(extract_subplot(image, r, c))

    rearranged_subplots = [subplots[i] for i in new_order]

    # Create a new grid for the rearranged subplots

    new_width = new_cols * subplot_width
    new_height = new_rows * subplot_height

    # Create a blank canvas for the new image
    new_image = Image.new('RGB', (new_width, new_height))

    # Paste subplots into the new image
    for idx, subplot in enumerate(rearranged_subplots):
        r, c = divmod(idx, new_cols)
        left = c * subplot_width
        upper = r * subplot_height
        new_image.paste(subplot, (left, upper))

    return new_image

def add_image_to_left(image, additional_image):
    
    # Ensure the additional image has the same height as the rearranged subplots
    # Resize the additional image while maintaining its aspect ratio
    new_height = image.height
    aspect_ratio = additional_image.width / additional_image.height
    new_width = int(new_height * aspect_ratio)
    additional_image = additional_image.resize((new_width, new_height))

    # Create a new blank canvas to combine the two images
    total_width = additional_image.width + image.width
    combined_image = Image.new('RGB', (total_width, new_height))

    # Paste the additional image on the left and the rearranged subplots on the right
    combined_image.paste(additional_image, (0, 0))
    combined_image.paste(image, (additional_image.width, 0))

    return combined_image

def zoom_image(image, zoom_factor):

    # Get image dimensions
    width, height = image.size

    # Calculate the size of the crop box
    crop_width = width // zoom_factor
    crop_height = height // zoom_factor

    # Calculate the coordinates for the crop box around the center
    left = (width - crop_width) // 2
    upper = (height - crop_height) // 2
    right = left + crop_width
    lower = upper + crop_height

    # Crop the image
    center_crop = image.crop((left, upper, right, lower))

    # Resize back to the original dimensions (optional, for zoom effect)
    return center_crop.resize((width, height), Image.Resampling.LANCZOS)

def stack_images(images, direction="vertical"):
    """
    Stacks multiple images into one image either vertically or horizontally.

    Args:
        image_paths (list): List of paths to the PNG files.
        output_file (str): Path to save the output stacked image.
        direction (str): "vertical" or "horizontal".
    """
    # Ensure all images have the same width (for vertical stacking)
    if direction == "vertical":
        widths = [img.width for img in images]
        target_width = min(widths)  # Resize to the smallest width for consistency
        images = [img.resize((target_width, int(img.height * target_width / img.width)), Image.Resampling.LANCZOS)
                  for img in images]
        total_width = target_width
        total_height = sum(img.height for img in images)
    else:  # Horizontal stacking
        heights = [img.height for img in images]
        target_height = min(heights)  # Resize to the smallest height for consistency
        images = [img.resize((int(img.width * target_height / img.height), target_height), Image.Resampling.LANCZOS)
                  for img in images]
        total_width = sum(img.width for img in images)
        total_height = target_height

    # Create a blank canvas for the stacked image
    stacked_img = Image.new("RGB", (total_width, total_height))

    # Paste each image onto the canvas
    offset = 0
    for img in images:
        if direction == "vertical":
            stacked_img.paste(img, (0, offset))
            offset += img.height
        else:  # Horizontal
            stacked_img.paste(img, (offset, 0))
            offset += img.width

    return stacked_img