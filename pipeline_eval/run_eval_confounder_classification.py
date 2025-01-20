import os
import yaml

from eval_confounder_classification import (
    build_gt_csv_from_seg_gt,
    evaluation_metrics_csv,
    count_errors,
    create_evaluation_yaml,
)

id_tags = ["0010", "0011", "0012"]

# Load the YAML file into a dictionary
dataset_yaml_file = "/workspace/home/data/datasets.yaml"
with open(dataset_yaml_file, "r") as file:
    datasets = yaml.safe_load(file)

# datasets
# get dataset
dataset_key = "ISIC2019"
dataset = datasets[dataset_key]

img_dir = dataset["raw_path"]
gt_dir = dataset["gt_seg_path"]

dataset_path = f"/workspace/home/data/{dataset_key}"

# csv file path
csv_file_gt = os.path.join(dataset_path, "gt_label.csv")
if not os.path.exists(csv_file_gt):
    build_gt_csv_from_seg_gt(csv_file_gt, img_dir, gt_dir)


for id_tag in id_tags:
    id_num = f"{id_tag}"
    base_dir = f"/workspace/home/pipeline_outputs/{datasets[dataset_key]['dataset_dir']}/{id_tag}"
    csv_file_pred = os.path.join(base_dir, "confounder_pred.csv")

    # evaluation metrics
    eval_output = evaluation_metrics_csv(csv_file_gt, csv_file_pred)
    num_errors = count_errors(csv_file_pred)

    # create yaml file with evaluation metrics
    yaml_file = os.path.join(base_dir, "confounder_detection_metrics.yaml")

    create_evaluation_yaml(yaml_file, eval_output, num_errors)
