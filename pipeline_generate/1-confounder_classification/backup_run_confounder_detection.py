import os
import sys
import yaml
import sglang as sgl
from sglang.lang.chat_template import get_chat_template
import multiprocessing as mp


# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import (
    get_images_from_directory,
    get_images_from_list,
    find_next_id,
    sort_csv_by_column,
    extract_index,
)
from pipeline_generate import build_pred_csv_sgl_with_backend, merge_csv_files

import argparse


parser = argparse.ArgumentParser(description="Perform confounder detection")
parser.add_argument(
    "-s",
    "--start",
    default=None,
    type=int,
    help="Image index to start from",
)

parser.add_argument(
    "--id",
    default=None,
    type=str,
    help="ID number for the output directory. If not given the next available ID will be used.",
)

parser.add_argument(
    "-nb",
    "--num_batches",
    default=1,
    type=int,
    help="Number of chunks to split the dataset into",
)


if __name__ == "__main__":

    args = parser.parse_args()

    # Load the YAML file into a dictionary
    dataset_yaml_file = "/workspace/home/data/datasets.yaml"
    with open(dataset_yaml_file, "r") as file:
        datasets = yaml.safe_load(file)

    # get dataset
    dataset_key = "ISIC2019"
    dataset = datasets[dataset_key]

    # load the ground truth, predicted segmentations and images
    images_path = dataset["raw_path"]

    if args.start is None:
        images = get_images_from_directory(images_path, without_img=True)
    else:
        # get list of image names from the directory
        image_names = os.listdir(images_path)
        # get the image names from the start index (the start index in the name)
        image_names = [
            name for name in image_names if extract_index(name) >= args.start
        ]
        # get the images from the list of image names
        images = get_images_from_list(image_names, images_path, without_img=True)

    # model and tokenizer paths for sgl server
    model_path = "liuhaotian/llava-v1.6-vicuna-7b"  # can I get it from backend?
    tokenizer_path = "llava-hf/llava-1.5-7b-hf"
    chat_template = "vicuna_v1.1"

    # model_path="lmms-lab/llama3-llava-next-8b",
    # tokenizer_path="lmms-lab/llama3-llava-next-8b-tokenizer",
    # chat_template = "llama-3-instruct"

    # generate output dir
    output_dir = (
        f"/workspace/home/pipeline_outputs/{datasets[dataset_key]['dataset_dir']}"
    )

    if args.id:
        id_num = args.id
    else:
        id_num = find_next_id(output_dir)

    output_dir += f"/{id_num}"

    os.makedirs(output_dir, exist_ok=True)

    # csv file path
    csv_file = f"{output_dir}/sgl_pred.csv"

    # check if file exists
    if os.path.exists(csv_file):
        # ask if the user wants to overwrite the file
        overwrite = input(
            f"The file {csv_file} already exists. Do you want to overwrite it? (y/n) "
        )
        if overwrite.lower() != "y":
            print("Exiting the program.")
            sys.exit()

    # Questions for (multi-turn) chat with model
    user_input_list = dataset["user_input_list"]

    # Model outputs that are considered as True
    output_for_true = dataset["output_for_true"]

    # interference
    build_pred_csv_sgl_with_backend(
        csv_file,
        images,
        user_input_list,
        output_for_true,
        num_batches=args.num_batches,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        chat_template=chat_template,
    )

    # csv_file = f"/workspace/home/pipeline_outputs/{datasets[dataset_key]['dataset_dir']}/0001/sgl_pred.csv"
    sort_csv_by_column(csv_file, column_index=0, header=True)

    # Generate meta data yaml file
    meta_data = {
        "id": id_num,
        "dataset": dataset_key,
        "model": model_path,
        "tokenizer": tokenizer_path,
        "chat_template": chat_template,
        "user_input_list": user_input_list,
        "output_for_true": output_for_true,
    }

    # Write the meta data to a YAML file
    meta_data_file = f"{output_dir}/meta_data.yaml"
    with open(meta_data_file, "w") as file:
        yaml.dump(meta_data, file)
