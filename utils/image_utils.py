from operator import is_
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def mask_on_image(image, segmentation_map):
    if isinstance(image, Image.Image):
        pil = True
        # Convert images to numpy arrays
        image_np = np.array(image)
        segmentation_map_np = np.array(segmentation_map)
    else:
        pil = False
        image_np = image
        segmentation_map_np = segmentation_map

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

    if pil:
        # Convert the result back to a PIL image
        return Image.fromarray(combined_image)
    else:
        return combined_image


def plot_grid(figures, label=None, ncols=3, title=None, save_path=None):
    # Define the number of columns for the grid
    num_rows = (
        len(figures) + ncols - 1
    ) // ncols  # Calculate the number of rows needed

    # Create a figure and a set of subplots
    if ncols > 4:
        fig, axs = plt.subplots(num_rows, ncols, figsize=(15, 2 * num_rows))
    else:
        fig, axs = plt.subplots(num_rows, ncols, figsize=(15, 5 * num_rows))

    # Flatten the axes array for easy iteration
    axs = axs.flatten()

    # Plot each figure in the grid
    for i, fig in enumerate(figures):
        axs[i].imshow(fig)
        axs[i].axis("off")  # Turn off the axis
        if label is not None:
            axs[i].set_title(label[i])

    # Turn off any remaining empty subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    if title is not None:
        plt.suptitle(title)

    if save_path is not None:
        plt.savefig(save_path)

    # Show the grid of figures
    plt.tight_layout()
    plt.show()


def dilate_mask(mask, kernel_size=5, iterations=1):
    is_pil = isinstance(mask, Image.Image)
    mask = np.array(mask, dtype=np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilate_mask = cv2.dilate(mask, kernel, iterations=iterations)
    if is_pil:
        return Image.fromarray(dilate_mask)
    else:
        return dilate_mask


def adjust_white_balance(tgt_img, ref_img):
    # Read the images
    tgt_img = np.array(tgt_img, dtype=np.float32)
    ref_img = np.array(ref_img, dtype=np.float32)

    if ref_img is None or tgt_img is None:
        raise ValueError("One or both of the images could not be loaded.")

    # Convert images from BGR to RGB (OpenCV uses BGR by default)
    ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    tgt_img_rgb = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)

    # Compute the mean color for each channel (R, G, B)
    ref_mean = np.mean(ref_img_rgb, axis=(0, 1))
    tgt_mean = np.mean(tgt_img_rgb, axis=(0, 1))

    # Compute the shift needed for each channel
    shift = ref_mean - tgt_mean

    # Apply the color shift to the target image
    tgt_img_rgb_shifted = tgt_img_rgb + shift
    tgt_img_rgb_shifted = np.clip(tgt_img_rgb_shifted, 0, 255).astype(np.uint8)

    # Convert the image back to BGR
    tgt_img_bgr_shifted = cv2.cvtColor(tgt_img_rgb_shifted, cv2.COLOR_RGB2BGR)

    # Convert the adjusted numpy array back to a PIL image
    adjusted_image = Image.fromarray(tgt_img_bgr_shifted.astype(np.uint8))

    return adjusted_image


def image_frequencies(image):
    # image to numpy
    image = np.array(image)

    # Split the image into its respective color channels
    channels = cv2.split(image)
    channel_colors = ["Blue", "Green", "Red"]
    channel_data = {}

    for idx, channel in enumerate(channels):
        unique, counts = np.unique(channel, return_counts=True)
        frequencies = dict(zip(unique, counts))
        channel_data[channel_colors[idx]] = frequencies

    # Plot the histograms for each channel
    plt.figure(figsize=(15, 5))
    for idx, (color, frequencies) in enumerate(channel_data.items()):
        plt.subplot(1, 3, idx + 1)
        plt.bar(
            frequencies.keys(),
            frequencies.values(),
            color=color.lower(),
            width=1.0,
            edgecolor="black",
        )
        plt.title(f"{color} Channel Pixel Intensity Frequency")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.xlim([0, 255])
    plt.tight_layout()
    plt.show()
