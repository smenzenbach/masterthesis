import os
import sys
import yaml


# Add the project root directory to sys.path
project_root = os.path.abspath("../..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import (
    create_file_names_list,
    get_images_from_directory,
    get_images_from_list,
    find_next_id,
    extract_index,
)
from confounder_classification_sgl import build_pred_csv_sgl_with_backend
from confounder_classification_gpt4 import build_pred_csv_gpt4

import argparse


parser = argparse.ArgumentParser(description="Perform confounder detection")

parser.add_argument(
    "-m",
    "--method",
    choices=["sgl", "gpt4"],
    required=True,
    help="Method to use for confounder detection",
)

parser.add_argument(
    "--id",
    default=None,
    type=str,
    help="ID tag for the output directory. If not given the next available ID will be used.",
)

parser.add_argument(
    "-s",
    "--start",
    default=None,
    type=int,
    help="Image index to start from",
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
        image_names = create_file_names_list(images_path)
        # get the image names from the start index (the start index in the name)
        image_names = [
            name for name in image_names if extract_index(name) >= args.start
        ]
        # get the images from the list of image names
        images = get_images_from_list(image_names, images_path, without_img=True)

    # set output dir
    output_dir = (
        f"/workspace/home/pipeline_outputs/{datasets[dataset_key]['dataset_dir']}"
    )

    if args.id:
        id_tag = args.id
    else:
        id_tag = find_next_id(output_dir)

    output_dir += f"/{id_tag}"

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

    # Model outputs that are considered as True
    output_for_true = dataset["output_for_true"]

    if args.method == "sgl":
        model_path = ("liuhaotian/llava-v1.6-vicuna-7b",)
        tokenizer_path = ("llava-hf/llava-1.5-7b-hf",)
        chat_template = ("vicuna_v1.1",)

        # Questions for (multi-turn) chat with model
        user_input_list = dataset["user_input_list"]

        # Build the prediction CSV using sgl
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

        # Generate meta data yaml file
        meta_data = {
            "id": id_tag,
            "dataset": dataset_key,
            "model": model_path,
            "tokenizer": tokenizer_path,
            "chat_template": chat_template,
            "user_input_list": user_input_list,
            "output_for_true": output_for_true,
        }

    elif args.method == "gpt4":
        prompt = dataset["prompt"]
        # TODO: maybe change to user_input_list[0] or something similar

        # Build the prediction CSV using gpt4
        build_pred_csv_gpt4(
            csv_file,
            images,
            prompt,
            output_for_true,
        )

        # Generate meta data yaml file
        meta_data = {
            "id": id_tag,
            "dataset": dataset_key,
            "prompt": prompt,
            "output_for_true": output_for_true,
        }

    # Write the meta data to a YAML file
    meta_data_file = f"{output_dir}/meta_data.yaml"
    with open(meta_data_file, "w") as file:
        yaml.dump(meta_data, file)
