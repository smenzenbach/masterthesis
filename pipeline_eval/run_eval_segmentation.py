import os, sys
import yaml
import toml


# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import get_images_from_directory

from eval_segmentation import write_seg_metrics_csv, write_avg_metrics_yaml_from_csv

id_tags = ["dito", "rmbg"]

for id_tag in id_tags:
    print(f"Running segmentation evaluation for {id_tag}...")
    # Load the configuration file
    config_path = f"/workspace/home01/pipeline_configs/config_{id_tag}.toml"
    config = toml.load(config_path)

    # Load YAML file into a dictionary
    dataset_name = config["dataset"]["name"]
    dataset_yaml_file = config["dataset"]["yaml_file"]
    with open(dataset_yaml_file, "r") as file:
        datasets = yaml.safe_load(file)

    # get dataset
    dataset = datasets[dataset_name]

    # get images
    if "use_existing" in config["segmentation"]:
        segmentation_id = config["segmentation"]["use_existing"]
        if segmentation_id == "gt":
            segmentation_path = dataset["gt_seg_path"]
        else:
            segmentation_path = (
                f"/workspace/home/pipeline_outputs/{dataset_name}/{segmentation_id}/seg"
            )
    else:
        segmentation_path = (
            f"/workspace/home/pipeline_outputs/{dataset_name}/{id_tag}/seg"
        )
        segmentation_id = id_tag

    segmentation_gt_path = datasets[dataset_name]["gt_seg_path"]

    segmentation_masks = get_images_from_directory(
        dir_path=segmentation_path
    )  # might be wrong: filter with label
    segmentation_masks_gt = get_images_from_directory(dir_path=segmentation_gt_path)

    segmentation_masks_hint_path = (
        "/workspace/home/data/ISIC2019/hint_data/ISIC2018_Task1_Training_GroundTruth"
    )
    segmentation_masks_hint = get_images_from_directory(segmentation_masks_hint_path)

    # create csv path
    base_dir = f"/workspace/home/pipeline_outputs/{dataset_name}/{id_tag}"
    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, "seg_metrics.csv")

    # create csv file

    if "inpainting" in config:
        dilation_kernel_size = config["inpainting"]["dilation"]["kernel_size"]
        dilation_iterations = config["inpainting"]["dilation"]["iterations"]
        if (
            "width" in config["inpainting"]["dilation"]
            and "height" in config["inpainting"]["dilation"]
        ):
            dilation_mask_width = config["inpainting"]["dilation"]["width"]
            dilation_mask_height = config["inpainting"]["dilation"]["height"]
            dilation_function_params = {
                "kernel_size": dilation_kernel_size,
                "iterations": dilation_iterations,
                "width": dilation_mask_width,
                "height": dilation_mask_height,
            }
        else:
            dilation_function_params = {
                "kernel_size": dilation_kernel_size,
                "iterations": dilation_iterations,
            }
    else:
        dilation_function_params = None
    print("start writing csv")
    write_seg_metrics_csv(
        csv_path,
        segmentation_masks,
        segmentation_masks_gt,
        segmentation_masks_hint,
        dilation_function_params,
    )

    print("start writing yaml")
    # create yaml file
    yaml_path = os.path.join(base_dir, "seg_metrics_avg.yaml")
    if not os.path.exists(yaml_path):
        write_avg_metrics_yaml_from_csv(csv_path, yaml_path)

    print(f"Segmentation metrics for {id_tag} written to {csv_path} and {yaml_path}.")
