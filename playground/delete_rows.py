import pandas as pd
import os, sys
import yaml

# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import (
    create_file_names_list,
    create_file_names_list_csv,
    get_completness_status,
    get_blocks,
    compare_file_lists,
)


def filter_csv_by_first_column(csv_file_path, filter_list, output_file_path):
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)

    # print number of rows
    print(f"Number of rows before filtering: {df.shape[0]}")

    # print first few rows
    print(df.head())

    # Filter rows where the first column's value is in the filter_list
    filtered_df = df[df[df.columns[0]].isin(filter_list)]

    # print number of rows
    print(f"Number of rows after filtering: {filtered_df.shape[0]}")

    # Write the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file_path, index=False)

    return filtered_df


# Load the YAML file into a dictionary
dataset_yaml_file = "/workspace/home/data/datasets.yaml"
with open(dataset_yaml_file, "r") as file:
    datasets = yaml.safe_load(file)

dataset_key = "ISIC2019"  # 'Pascal_VOC07' #
dataset = datasets[dataset_key]


id_num = "0010"
dataset_key = "ISIC2019"

gt_csv_path = f"/workspace/home/data/{dataset_key}/gt_label.csv"
pred_csv_path = f"/workspace/home/pipeline_outputs/{dataset_key}/{id_num}/sgl_pred.csv"

gt_files = create_file_names_list_csv(gt_csv_path)
pred_files = create_file_names_list_csv(pred_csv_path)

missing_files, copied_files = compare_file_lists(pred_files, gt_files)

output_dir = (
    f"/workspace/home/pipeline_outputs/{datasets[dataset_key]['dataset_dir']}/0010"
)
csv_source_path = f"{output_dir}/sgl_pred5.csv"
csv_target_path = f"{output_dir}/sgl_pred6.csv"

filter_csv_by_first_column(csv_source_path, missing_files, csv_target_path)
