import os
import sys
import yaml
import toml

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


parser = argparse.ArgumentParser(description="Perform confounder classification")

parser.add_argument(
    "--config", required=True, type=str, help="Path to the configuration file"
)

parser.add_argument(
    "-s",
    "--start",
    default=None,
    type=int,
    help="Image index to start from",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Load the configuration file
    config = toml.load(args.config)

    # check if existing confounder classificaton should be used
    if "use_existing" in config["confounder_classification"]:
        print("Using existing confounder classification. Exiting the program.")
        sys.exit()

    # get id that is between "_" and "." in the config file name
    id_tag = int(args.config.split("_")[-1].split(".")[0])

    # Load YAML file into a dictionary
    dataset_name = config["dataset"]["name"]
    dataset_yaml_file = config["dataset"]["yaml_file"]
    with open(dataset_yaml_file, "r") as file:
        datasets = yaml.safe_load(file)

    # get dataset
    dataset = datasets[dataset_name]

    # get images
    images_path = dataset["raw_path"]

    if args.start is None:
        images = get_images_from_directory(images_path)
    else:
        # get list of image names from the directory
        image_names = create_file_names_list(images_path)
        # get the image names from the start index (the start index in the name)
        image_names = [
            name for name in image_names if extract_index(name) >= args.start
        ]
        # get the images from the list of image names
        images = get_images_from_list(image_names, images_path)

    # set output dir
    output_dir = f"/workspace/home/pipeline_outputs/{config["dataset"]["name"]}/{id_tag}"

    os.makedirs(output_dir, exist_ok=True)

    # csv file path
    csv_file = f"{output_dir}/confounder_pred.csv"

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
    output_for_true = config["confounder_classification"]["output_for_true"]

    # retrieve used model
    model = config["confounder_classification"]["model"]

    if model == "sgl":
        model_path = config["confounder_classification.sgl"]["model_path"]
        tokenizer_path = config["confounder_classification.sgl"]["tokenizer_path"]
        chat_template = config["confounder_classification.sgl"]["chat_template"]

        # Questions for (multi-turn) chat with model
        user_input_list = config["confounder_classification.sgl"]["user_input_list"]

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

    elif model == "gpt4":
        prompt = config["confounder_classification.gpt4"]["prompt"]
        # TODO: maybe change to user_input_list[0] or something similar

        # Build the prediction CSV using gpt4
        build_pred_csv_gpt4(
            csv_file,
            images,
            prompt,
            output_for_true,
        )

        # Generate meta data yaml file

    # Write the meta data to a YAML file
    meta_data_file = f"{output_dir}/meta_data.yaml"
    with open(meta_data_file, "w") as file:
        yaml.dump(meta_data, file)
