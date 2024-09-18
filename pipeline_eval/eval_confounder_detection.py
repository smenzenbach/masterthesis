import os, sys
import csv
import yaml
import warnings

from dataclasses import dataclass

# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import load_images, get_images_from_directory


def build_gt_csv_from_seperated_folders(
    csv_path, images_dir_path, confounder_dir, no_confounder_dir
):
    """
    Build a csv file with ground truth values if a confounder is in the image. 1 if confounder is present, 0 otherwise.
    It is assumed that the confounder is present in the images in the confounder_dir and not present in the images in the no_confounder_dir.

    The following data structure is expected:
    images_dir_path/
        confounder_dir/
            image1.jpg
            image2.jpg
            ...
        no_confounder_dir/
            image3.jpg
            image4.jpg
            ...

    Parameters
    ----------
    csv_path : str
        The path to save the csv file.
    images_dir_path : str
        The path to the directory of the images of the dataset.
    confounder_dir : str
        The name of the directory where the confounder is present.
    no_confounder_dir : str
        The name of the directory where the confounder is not present.

    Returns
    -------
    None
    """
    images_confounder, images_no_confounder = load_images(
        dir_path=images_dir_path,
        confounder_dir=confounder_dir,
        no_confounder_dir=no_confounder_dir,
    )

    with open(csv_path, mode="w") as csv_file:
        writer = csv.writer(csv_file)

        # Write header
        writer.writerow(["image_file_name", "label"])

        def write_to_csv(images, label):
            for img_confounder_dict in images:
                image_file_name = img_confounder_dict["meta"]["file_name"]
                writer.writerow([image_file_name, label])

        write_to_csv(images_confounder, 1)
        write_to_csv(images_no_confounder, 0)


def build_gt_csv_from_seg_gt(csv_path, images_dir_path, seg_dir_path):
    """
    Build a csv file with ground truth values if a confounder is in the image. 1 if confounder is present, 0 otherwise.
    It is assumed that the confounder is present in the images in the seg_dir_path.

    Parameters
    ----------
    csv_path : str
        The path to save the csv file.
    images_dir_path : str
        The path to the directory of the images of the dataset.
    seg_dir_path : str
        The path to the directory where the segmented images are stored.

    Returns
    -------
    None
    """

    images = get_images_from_directory(images_dir_path, without_img=True, sub_dir=True)
    seg_images = get_images_from_directory(seg_dir_path, without_img=True, sub_dir=True)

    seg_images = [img["file_name"] for img in seg_images]

    with open(csv_path, mode="w") as csv_file:
        writer = csv.writer(csv_file)

        # Write header
        writer.writerow(["image_file_name", "label"])

        for img_dict in images:
            image_file_name = img_dict["file_name"]
            label = 1 if image_file_name in seg_images else 0
            writer.writerow([image_file_name, label])


def evaluation_metrics(gt_csv, predictions_csv):
    """
    Compare the ground truth values with the predictions and return the evaluation metrics: accuracy, recall, precision, and f1.
    It is assumed that the csv files have a header.


    You can use the output like this:
    ```
    metrics = compare_gt_with_predictions("gt.csv", "predictions.csv")
    accuracy = metrics.accuracy
    ```

    Parameters
    ----------
    gt_csv : str
        The path to the ground truth csv file.
    predictions_csv : str
        The path to the predictions csv file.

    Returns
    -------
    EvaluationMetrics
        A dataclass with the evaluation metrics: accuracy, recall, precision, and f1.
        - accuracy: Measures the proportion of true results among the total number of cases examined.
        - recall: Measures the proportion of actual positives that are correctly identified.
        - precision: Measures the proportion of positive identifications that were actually correct.
        - f1: The harmonic mean of precision and recall.
        - false_positive: The number of false positives.
        - false_negative: The number of false negatives.
        - true_positive: The number of true positives.
        - true_negative: The number of true negatives.
        - num_imgs: The number of images in the predictions csv file.

    Warnings
    --------
    If there are images in the ground truth csv that are not in the predictions csv, a warning will be raised.
    """

    # dataclass
    @dataclass
    class EvaluationMetrics:
        accuracy: float
        recall: float
        precision: float
        f1: float
        false_positive: int = 0
        false_negative: int = 0
        true_positive: int = 0
        true_negative: int = 0
        num_imgs: int = 0

    with open(gt_csv, mode="r") as gt_file, open(
        predictions_csv, mode="r"
    ) as pred_file:
        gt_reader = csv.reader(gt_file)
        pred_reader = csv.reader(pred_file)

        # Skip header
        next(gt_reader)
        next(pred_reader)

        gt_dict = {row[0]: int(row[1]) for row in gt_reader}
        pred_dict = {row[0]: int(row[1]) for row in pred_reader}

        false_positive = 0
        false_negative = 0
        true_positive = 0
        true_negative = 0

        for img_name, pred_label in pred_dict.items():
            gt_label = gt_dict.get(img_name, None)
            if gt_label is not None and pred_label == 1:
                if gt_label == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            elif gt_label is not None and pred_label == 0:
                if gt_label == 0:
                    true_negative += 1
                else:
                    false_negative += 1

        accuracy = (true_positive + true_negative) / (
            true_positive + true_negative + false_positive + false_negative
        )
        recall = true_positive / (true_positive + false_negative)
        precision = true_positive / (true_positive + false_positive)
        f1 = 2 * (precision * recall) / (precision + recall)

        num_imgs = len(pred_dict)

        # check if there are any images in the ground truth csv that are not in the predictions csv
        if len(gt_dict) > len(pred_dict):
            missing_imgs = set(gt_dict.keys()) - set(pred_dict.keys())
            warnings.warn(
                f"The following {len(missing_imgs)} images are missing in the predictions csv: {sorted(missing_imgs)}"
            )

        return EvaluationMetrics(
            accuracy,
            recall,
            precision,
            f1,
            false_positive,
            false_negative,
            true_positive,
            true_negative,
            num_imgs,
        )


def create_evaluation_yaml(yaml_path, metrics):
    """
    Create a yaml file with the evaluation metrics.

    Parameters
    ----------
    yaml_path : str
        The path to save the yaml file.
    metrics : EvaluationMetrics
        A dataclass with the evaluation metrics.

    Returns
    -------
    None
    """

    data = {
        "accuracy": metrics.accuracy,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "f1": metrics.f1,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "true_positive": metrics.true_positive,
        "true_negative": metrics.true_negative,
        "num_imgs": metrics.num_imgs,
    }

    with open(yaml_path, "w") as yaml_file:
        yaml.dump(data, yaml_file)
