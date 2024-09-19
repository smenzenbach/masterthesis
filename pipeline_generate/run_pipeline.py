import os
import sys
import csv
import yaml
import toml

# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import get_images_from_directory, get_images_from_list

import argparse

parser = argparse.ArgumentParser(description="Pipeline")


parser.add_argument(
    "--config",
    required=True,
    type=str,
    help="Path to the configuration file",
)
args = parser.parse_args()

#########################################
#                                       #
#    Load config and extract id         #
#                                       #
#########################################
config = toml.load(args.config)

# get id that is between "_" and "."
id_tag = int(args.config.split("_")[-1].split(".")[0])


#########################################
#                                       #
#             Load dataset              #
#                                       #
#########################################

# Load the YAML file into a dictionary
dataset_name = config["dataset"]["name"]
dataset_yaml_file = config["dataset"]["yaml_file"]
with open(dataset_yaml_file, "r") as file:
    datasets = yaml.safe_load(file)

dataset = datasets[dataset_name]

#########################################
#                                       #
#                run vlm                #
#                                       #
#########################################

raw_images = get_images_from_directory(dataset["raw_path"])

if config["segmentation"]["model"] == "vlm":
    raise NotImplementedError("VLM segmentation is not implemented yet")
else:
    num_batches = config["confounder_detection"]["num_batches"]
    model_path = config["confounder_detection"]["model_path"]
    tokenizer_path = config["confounder_detection"]["tokenizer_path"]
    output_path = config["confounder_detection"]["output_path"]


#########################################
#                                       #
#           run segmentation            #
#                                       #
#########################################
