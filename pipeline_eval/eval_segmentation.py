import os, sys
import cv2
from httpx import head
import numpy as np
import warnings
import csv
from PIL import Image

# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from sympy import per
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


def percentage_hint_on_mask(mask, mask_hint):
    """
    Returns the percentage of pixels of the hint mask that are part of the mask.
    Same as percentage_gt_on_mask but for hint mask.
    """
    return percentage_gt_on_mask(mask, mask_hint)


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


# create hint masks (segmentation mask with circle in the center)
def create_hint_mask(image_shape):
    hint_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    center = (image_shape[1] // 2, image_shape[0] // 2)
    radius = min(image_shape[0], image_shape[1]) // 12  # Adjust the radius as needed
    y, x = np.ogrid[: image_shape[0], : image_shape[1]]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
    hint_mask[mask] = 255
    return hint_mask


def get_img_name_from_hint(hint_img_name):
    # from ISIC_0000000_segmentation.png to ISIC_0000000.jpg
    hint_img_name = hint_img_name.split("_")[1]
    hint_img_name = hint_img_name.split(".")[0]
    hint_img_name = "ISIC_" + hint_img_name + ".jpg"
    return hint_img_name


def write_seg_metrics_csv(
    csv_path, masks, gt_masks, masks_hints=None, dilation_function_params=None
):
    """
    Generate a csv file with the segmentation evaluation metrics: IoU, Dice coefficient, pixel accuracy, percentage of mask on ground truth, and percentage of ground truth on mask.

    Parameters
    ----------
    csv_patch : str
        The path to save the csv file.
    masks : output of get_images.. function
        list of dictionaries with the image file name and path of the segmentation masks.
    gt_masks : output of get_images.. function
        list of dictionaries with the image file name and path of the ground truth masks.
    masks_hints : output of get_images.. function, optional
        list of dictionaries with the image file name and path of the hint masks. The default is None.
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
        header = [
            "image_file_name",
            "iou",
            "dice_coefficient",
            "pixel_accuracy",
            "percentage_mask_on_gt",
            "percentage_gt_on_mask",
            "dilation_kernel_size",
            "dilation_iterations",
        ]

        if masks_hints is not None:
            header.append(["has_hint_mask", "percentage_hint_on_mask"])
            hint_mask = create_hint_mask(
                np.array(Image.open(masks[0]["file_path"])).shape[:2]
            )
            masks_hint_names = [
                get_img_name_from_hint(hint_dict["file_name"])
                for hint_dict in masks_hints
            ]

        writer.writerow(header)

        for mask_dict, mask_gt_dict in zip(masks, gt_masks):
            image_file_name = mask_dict["file_name"]

            mask_url = mask_dict["file_path"]
            mask = Image.open(mask_url).convert("L")

            mask_gt_url = mask_gt_dict["file_path"]
            mask_gt = Image.open(mask_gt_url).convert("L")

            # adapt mask and mask_gt size to same size (size of mask)
            mask_gt = mask_gt.resize(mask.size)

            mask_np = np.array(mask)
            mask_gt_np = np.array(mask_gt)

            # assure binary masks
            mask_np = np.where(mask_np > 128, 255, 0)
            mask_gt_np = np.where(mask_gt_np > 128, 255, 0)

            if dilation_function_params is not None:
                mask_np = dilate_mask(mask_np, **dilation_function_params)

            iou = iou_score(mask_np, mask_gt_np)
            dice = dice_coefficient(mask_np, mask_gt_np)
            pixel_acc = pixel_accuracy(mask_np, mask_gt_np)
            mask_on_gt = percentage_mask_on_gt(mask_np, mask_gt_np)
            gt_on_mask = percentage_gt_on_mask(mask_np, mask_gt_np)
            if masks_hints is not None:
                if image_file_name in masks_hint_names:
                    hint_mask_path = masks_hints[
                        masks_hint_names.index(image_file_name)
                    ]["file_path"]
                    hint_mask = np.array(
                        Image.open(hint_mask_path)
                        .convert("L")
                        .point(lambda x: 255 if x > 128 else 0)
                    )
                else:
                    hint_mask = hint_mask
                hint_on_mask = percentage_hint_on_mask(mask_np, hint_mask)

            metrics_row = [
                image_file_name,
                iou,
                dice,
                pixel_acc,
                mask_on_gt,
                gt_on_mask,
                dilation_function_params["kernel_size"],
                dilation_function_params["iterations"],
            ]

            if masks_hints is not None:
                metrics_row.append(1 if image_file_name in masks_hint_names else 0)
                metrics_row.append(hint_on_mask)

            writer.writerow(metrics_row)


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

        dilation_kernel_size = None
        dilation_iterations = None

        has_hint_mask_list = []
        hint_on_mask_list = []

        for row in reader:
            iou_list.append(float(row[1]))
            dice_list.append(float(row[2]))
            pixel_acc_list.append(float(row[3]))
            mask_on_gt_list.append(float(row[4]))
            gt_on_mask_list.append(float(row[5]))
            if dilation_kernel_size is None:
                dilation_kernel_size = row[6]
                dilation_iterations = row[7]
            if len(row) > 8:
                has_hint_mask_list.append(int(row[8]))
                hint_on_mask_list.append(float(row[9]))

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

        if len(row) > 8:
            hint_on_mask_array = np.array(hint_on_mask_list)
            num_imgs_with_hint = np.sum(has_hint_mask_list)
            mean_hint_on_mask = np.mean(hint_on_mask_array)
            std_hint_on_mask = np.std(hint_on_mask_array)
            median_hint_on_mask = np.median(hint_on_mask_array)
            mean_hint_on_mask_over_zero = np.mean(
                hint_on_mask_array[hint_on_mask_array > 0]
            )
            std_hint_on_mask_over_zero = np.std(
                hint_on_mask_array[hint_on_mask_array > 0]
            )
            median_hint_on_mask_over_zero = np.median(
                hint_on_mask_array[hint_on_mask_array > 0]
            )
            number_hint_filled = np.sum(hint_on_mask_array > 0)
            number_hint_filled_over_p1 = np.sum(hint_on_mask_array > 0.1)
            number_hint_filled_over_p25 = np.sum(hint_on_mask_array > 0.25)
            number_hint_filled_over_p5 = np.sum(hint_on_mask_array > 0.5)

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
                f"dilation_kernel_size: {dilation_kernel_size}\n"
                f"dilation_iterations: {dilation_iterations}\n"
                f"n_images: {len(iou_list)}\n"
            )
            if len(row) > 8:
                yaml_file.write(
                    f"mean_hint_on_mask: {mean_hint_on_mask:.5f}\n"
                    f"std_hint_on_mask: {std_hint_on_mask:.5f}\n"
                    f"median_hint_on_mask: {median_hint_on_mask:.5f}\n"
                    f"mean_hint_on_mask_over_zero: {mean_hint_on_mask_over_zero:.3f}\n"
                    f"std_hint_on_mask_over_zero: {std_hint_on_mask_over_zero:.3f}\n"
                    f"median_hint_on_mask_over_zero: {median_hint_on_mask_over_zero:.3f}\n"
                    f"n_images_with_hint_mask: {num_imgs_with_hint}\n"
                    f"n_images_hint_filled: {number_hint_filled}\n"
                    f"n_images_hint_filled_over_p1: {number_hint_filled_over_p1}\n"
                    f"n_images_hint_filled_over_p25: {number_hint_filled_over_p25}\n"
                    f"n_images_hint_filled_over_p5: {number_hint_filled_over_p5}\n"
                )
