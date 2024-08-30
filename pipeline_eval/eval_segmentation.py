import os, sys
import cv2
import numpy as np
import warnings
import csv
from PIL import Image

# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import dilate_mask


def iou_score(mask, mask_gt):
    """
    Intersection over Union (IoU) metric for segmentation evaluation.

    Parameters
    ----------
    mask : np.ndarray
        The predicted mask.
    mask_gt : np.ndarray
        The ground truth mask.

    Returns
    -------
    float
        The IoU score.

    Warnings
    --------
    If the masks have different shapes, a warning is raised and the mask is resized to the ground truth shape.
    """
    # make sure the segmentation is binary
    if np.unique(mask).size > 2 or np.unique(mask_gt).size > 2:
        raise ValueError("The segmentation masks must be binary.")

    # Check if the masks have the same shape and adjust if necessary
    if mask.shape != mask_gt.shape:
        warnings.warn(
            "The masks have different shapes. Resizing the mask to the ground truth shape."
        )
        mask = cv2.resize(mask.astype(float), mask_gt.shape[::-1])

    # Compute the intersection and union
    intersection = np.logical_and(mask, mask_gt).sum()
    union = np.logical_or(mask, mask_gt).sum()

    # Compute the IoU
    iou = intersection / union
    return iou


def dice_coefficient(mask, mask_gt):
    """
    Dice coefficient metric for segmentation evaluation.

    Parameters
    ----------
    mask : np.ndarray
        The predicted mask.
    mask_gt : np.ndarray
        The ground truth mask.

    Returns
    -------
    float
        The Dice coefficient score.

    Warnings
    --------
    If the masks have different shapes, a warning is raised and the mask is resized to the ground truth shape.
    """
    # make sure the segmentation is binary
    if np.unique(mask).size > 2 or np.unique(mask_gt).size > 2:
        raise ValueError("The segmentation masks must be binary.")

    # Check if the masks have the same shape and adjust if necessary
    if mask.shape != mask_gt.shape:
        warnings.warn(
            "The masks have different shapes. Resizing the mask to the ground truth shape."
        )
        mask = cv2.resize(mask.astype(float), mask_gt.shape[::-1])

    # Compute the intersection
    intersection = np.logical_and(mask, mask_gt).sum()

    # Compute the Dice coefficient
    dice = 2 * intersection / (mask.sum() + mask_gt.sum())
    return dice


def pixel_accuracy(mask, mask_gt):
    """
    Pixel accuracy metric for segmentation evaluation.

    Parameters
    ----------
    mask : np.ndarray
        The predicted mask.
    mask_gt : np.ndarray
        The ground truth mask.

    Returns
    -------
    float
        The pixel accuracy score.

    Warnings
    --------
    If the masks have different shapes, a warning is raised and the mask is resized to the ground truth shape.
    """
    # make sure the segmentation is binary
    if np.unique(mask).size > 2 or np.unique(mask_gt).size > 2:
        raise ValueError("The segmentation masks must be binary.")

    # Check if the masks have the same shape and adjust if necessary
    if mask.shape != mask_gt.shape:
        warnings.warn(
            "The masks have different shapes. Resizing the mask to the ground truth shape."
        )
        mask = cv2.resize(mask.astype(float), mask_gt.shape[::-1])

    # Compute the pixel accuracy
    pixel_acc = (mask == mask_gt).sum() / mask.size
    return pixel_acc


def percentage_mask_on_gt(mask, mask_gt):
    """
    Returns the percentage of pixels of the mask that are part of the ground truth segmentation.

    Parameters
    ----------
    mask : np.ndarray
        The predicted mask.
    mask_gt : np.ndarray
        The ground truth mask.

    Returns
    -------
    float
        The percentage of pixels of the mask that are part of the ground truth segmentation.

    Warnings
    --------
    If the masks have different shapes, a warning is raised and the mask is resized to the ground truth shape.
    """

    # make sure the segmentation is binary
    if np.unique(mask).size > 2 or np.unique(mask_gt).size > 2:
        raise ValueError("The segmentation masks must be binary.")

    # Check if the masks have the same shape and adjust if necessary
    if mask.shape != mask_gt.shape:
        warnings.warn(
            "The masks have different shapes. Resizing the mask to the ground truth shape."
        )
        mask = cv2.resize(mask.astype(float), mask_gt.shape[::-1])

    # Compute the percentage of pixels of the mask that are part of the ground truth segmentation
    if np.sum(mask) == 0:
        return 0
    else:
        percentage = np.sum(np.logical_and(mask, mask_gt)) / np.sum(mask)
        return percentage


def percentage_gt_on_mask(mask, mask_gt):
    """
    Returns the percentage of pixels of the ground truth segmentation that are part of the mask.

    Parameters
    ----------
    mask : np.ndarray
        The predicted mask.
    mask_gt : np.ndarray
        The ground truth mask.

    Returns
    -------
    float
        The percentage of pixels of the ground truth segmentation that are part of the mask.

    Warnings
    --------
    If the masks have different shapes, a warning is raised and the mask is resized to the ground truth shape.
    """

    # make sure the segmentation is binary
    if np.unique(mask).size > 2 or np.unique(mask_gt).size > 2:
        raise ValueError("The segmentation masks must be binary.")

    # Check if the masks have the same shape and adjust if necessary
    if mask.shape != mask_gt.shape:
        warnings.warn(
            "The masks have different shapes. Resizing the mask to the ground truth shape."
        )
        mask = cv2.resize(mask.astype(float), mask_gt.shape[::-1])

    # Compute the percentage of pixels of the ground truth segmentation that are part of the mask
    if np.sum(mask_gt) == 0:
        return 0
    else:
        percentage = np.sum(np.logical_and(mask, mask_gt)) / np.sum(mask_gt)
        return percentage


def seg_percentage(gt_segmentation):
    """
    Returns the percentage of pixels that are part of the segmentation.

    Parameters
    ----------
    gt_segmentation : np.ndarray
        The ground truth segmentation mask.

    Returns
    -------
    float
        The percentage of pixels that are part of the segmentation.
    """
    # make sure the segmentation is binary
    if np.unique(gt_segmentation).size > 2:
        raise ValueError("The segmentation masks must be binary.")

    return np.sum(gt_segmentation) / gt_segmentation.size


def write_seg_metrics_csv(csv_path, masks, gt_masks, dilation_function_params=None):
    """
    Generate a csv file with the segmentation evaluation metrics: IoU, Dice coefficient, pixel accuracy, percentage of mask on ground truth, and percentage of ground truth on mask.

    Parameters
    ----------
    csv_patch : str
        The path to save the csv file.
    masks : output of load_images
        A list of dictionaries containing the image objects and their metadata.
    gt_masks : output of load_images
        A list of dictionaries containing the image objects and their metadata.
    dilation_function_params : dict, optional
        The parameters to pass to the dilation function. The default is None.
        {'kernel_size': dilation_kernel_size, 'iterations': dilation_iterations}


    Returns
    -------
    None
    """
    with open(csv_path, mode="w") as csv_file:
        writer = csv.writer(csv_file)

        # Write header
        writer.writerow(
            [
                "image_file_name",
                "iou",
                "dice_coefficient",
                "pixel_accuracy",
                "percentage_mask_on_gt",
                "percentage_gt_on_mask",
                "dilation_kernel_size",
                "dilation_iterations",
            ]
        )

        for mask_dict, mask_gt_dict in zip(masks, gt_masks):
            image_file_name = mask_dict["meta"]["file_name"]

            mask_url = mask_dict["meta"]["file_path"]
            mask = Image.open(mask_url).convert("L")

            mask_gt_url = mask_gt_dict["meta"]["file_path"]
            mask_gt = Image.open(mask_gt_url).convert("L")

            # adapt mask and mask_gt size to same size (size of mask)
            mask_gt = mask_gt.resize(mask.size)

            mask_np = np.array(mask)
            mask_gt_np = np.array(mask_gt)

            # assure binary masks
            mask_np = np.where(mask_np > 128, 1, 0)
            mask_gt_np = np.where(mask_gt_np > 128, 1, 0)

            if dilation_function_params is not None:
                mask_np = dilate_mask(mask_np, **dilation_function_params)

            iou = iou_score(mask_np, mask_gt_np)
            dice = dice_coefficient(mask_np, mask_gt_np)
            pixel_acc = pixel_accuracy(mask_np, mask_gt_np)
            mask_on_gt = percentage_mask_on_gt(mask_np, mask_gt_np)
            gt_on_mask = percentage_gt_on_mask(mask_np, mask_gt_np)

            writer.writerow(
                [
                    image_file_name,
                    iou,
                    dice,
                    pixel_acc,
                    mask_on_gt,
                    gt_on_mask,
                    dilation_function_params["kernel_size"],
                    dilation_function_params["iterations"],
                ]
            )


def write_avg_metrics_yaml_from_csv(csv_path, yaml_path):
    """
    Generate a yaml file with the average segmentation evaluation metrics: IoU, Dice coefficient, pixel accuracy, percentage of mask on ground truth, and percentage of ground truth on mask.
    It is assumed that the csv file has the following columns: iou, dice_coefficient, pixel_accuracy, percentage_mask_on_gt, percentage_gt_on_mask and the first row is the header.

    Parameters
    ----------
    csv_path : str
        The path to the csv file with the segmentation evaluation metrics.
    yaml_path : str
        The path to save the yaml file.

    Returns
    -------
    None
    """
    with open(csv_path, mode="r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # skip header

        iou_list = []
        dice_list = []
        pixel_acc_list = []
        mask_on_gt_list = []
        gt_on_mask_list = []

        for row in reader:
            iou_list.append(float(row[1]))
            dice_list.append(float(row[2]))
            pixel_acc_list.append(float(row[3]))
            mask_on_gt_list.append(float(row[4]))
            gt_on_mask_list.append(float(row[5]))

        # Compute the average metrics
        mean_iou = np.mean(iou_list)
        mean_dice = np.mean(dice_list)
        mean_pixel_acc = np.mean(pixel_acc_list)
        mean_mask_on_gt = np.mean(mask_on_gt_list)
        mean_gt_on_mask = np.mean(gt_on_mask_list)

        # Compute standard deviation
        std_iou = np.std(iou_list)
        std_dice = np.std(dice_list)
        std_pixel_acc = np.std(pixel_acc_list)
        std_mask_on_gt = np.std(mask_on_gt_list)
        std_gt_on_mask = np.std(gt_on_mask_list)

        # Compute median
        median_iou = np.median(iou_list)
        median_dice = np.median(dice_list)
        median_pixel_acc = np.median(pixel_acc_list)
        median_mask_on_gt = np.median(mask_on_gt_list)
        median_gt_on_mask = np.median(gt_on_mask_list)

        with open(yaml_path, "w") as yaml_file:
            yaml_file.write(
                f"mean_iou: {mean_iou:.3f}\n"
                f"std_iou: {std_iou:.3f}\n"
                f"median_iou: {median_iou:.3f}\n"
                f"mean_dice: {mean_dice:.3f}\n"
                f"std_dice: {std_dice:.3f}\n"
                f"median_dice: {median_dice:.3f}\n"
                f"mean_pixel_accuracy: {mean_pixel_acc:.3f}\n"
                f"std_pixel_accuracy: {std_pixel_acc:.3f}\n"
                f"median_pixel_accuracy: {median_pixel_acc:.3f}\n"
                f"mean_percentage_mask_on_gt: {mean_mask_on_gt:.3f}\n"
                f"std_percentage_mask_on_gt: {std_mask_on_gt:.3f}\n"
                f"median_percentage_mask_on_gt: {median_mask_on_gt:.3f}\n"
                f"mean_percentage_gt_on_mask: {mean_gt_on_mask:.3f}\n"
                f"std_percentage_gt_on_mask: {std_gt_on_mask:.3f}\n"
                f"median_percentage_gt_on_mask: {median_gt_on_mask:.3f}\n"
                f"dilation_kernel_size: {row[6]}\n"
                f"dilation_iterations: {row[7]}\n"
                f"n_images: {len(iou_list)}"
            )
