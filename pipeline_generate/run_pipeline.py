import os
import sys
import csv
import yaml

# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import get_images_from_directory, get_images_from_list

import argparse

parser = argparse.ArgumentParser(description="Pipeline")

parser.add_argument(
    "--dataset",
    default="ISIC2019",
    type=str,
    choices=["ISIC2019", "Pascal_VOC07"],
    help="Dataset to use",
)

parser.add_argument(
    "--id",
    default=None,
    type=str,
    help="ID number for the output directory. If not given the next available ID will be used.",
)
