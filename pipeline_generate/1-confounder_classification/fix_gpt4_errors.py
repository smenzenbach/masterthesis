import os
import sys
import yaml
import toml

# Add the project root directory to sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)  # Adjust this path if needed
if project_root not in sys.path:
    sys.path.append(project_root)


from utils import (
    create_file_names_list,
    get_images_from_directory,
    get_images_from_list,
    extract_index,
)

from confounder_classification_gpt4 import fix_errors_in_csv

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
        print(
            f"Using existing confounder classification from {config['confounder_classification']['use_existing']}. Exiting the program."
        )
        sys.exit()

    # get id that is between the first "_" and "." in the config file name
    id_tag = args.config.split("/")[-1].split("_", 1)[1].split(".")[0]

    # Load YAML file into a dictionary
    dataset_name = config["dataset"]["name"]
    dataset_yaml_file = config["dataset"]["yaml_file"]
    with open(dataset_yaml_file, "r") as file:
        datasets = yaml.safe_load(file)

    # get dataset
    dataset_key = "ISIC2019"
    dataset = datasets[dataset_key]

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
    output_dir = (
        f"/workspace/home/pipeline_outputs/{config['dataset']['name']}/{id_tag}"
    )

    os.makedirs(output_dir, exist_ok=True)

    # csv file path
    csv_file = f"{output_dir}/confounder_pred.csv"

    # check if file exists
    if not os.path.exists(csv_file):
        print(f"File {csv_file} does not exist. Exiting the program.")
        sys.exit()

    # Model outputs that are considered as True
    output_for_true = config["confounder_classification"]["output_for_true"]

    # retrieve used model
    model_name = config["confounder_classification"]["model"]

    if "gpt" not in model_name:
        print(
            f"Model {model_name} is not supported. Has to be a gpt model. Exiting the program."
        )
        sys.exit()

    print(f"Start fixing confounder classification with model: {model_name}")

    prompt = config["confounder_classification"][model_name]["prompt"]
    # TODO: maybe change to user_input_list[0] or something similar

    # Build the prediction CSV using gpt4
    fix_errors_in_csv(
        csv_file,
        images,
        prompt,
        output_for_true,
    )

    print(f"Finished fixing confounder classification with model: {model_name}")

    # Generate meta data yaml file

    # Write the meta data to a YAML file
    # meta_data_file = f"{output_dir}/meta_data.yaml"
    # with open(meta_data_file, "w") as file:
    #    yaml.dump(meta_data, file)
