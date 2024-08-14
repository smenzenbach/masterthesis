import numpy as np
import cv2
from PIL import Image


def mask_on_image(image, segmentation_map):
    # Convert images to numpy arrays
    image_np = np.array(image)
    segmentation_map_np = np.array(segmentation_map)

    # Ensure the segmentation map is in grayscale
    if len(segmentation_map_np.shape) == 3 and segmentation_map_np.shape[2] == 3:
        segmentation_map_np = cv2.cvtColor(segmentation_map_np, cv2.COLOR_RGB2GRAY)

    # Ensure the images have the same shape
    if image_np.shape[:2] != segmentation_map_np.shape[:2]:
        raise ValueError(
            "The base image and segmentation map must have the same dimensions."
        )

    # Create a colored overlay for the segmentation map
    colored_overlay = np.zeros_like(image_np)
    colored_overlay[segmentation_map_np == 255] = [
        255,
        0,
        0,
    ]  # Red color for the segmentation

    # Combine the images with some transparency for the overlay
    alpha = 0.5  # Transparency factor
    combined_image = cv2.addWeighted(image_np, 1, colored_overlay, alpha, 0)

    # Convert the result back to a PIL image
    return Image.fromarray(combined_image)
