import os, sys
from regex import P
from tqdm import tqdm
import yaml
import toml
import argparse
import numpy as np


# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import (
    get_images_from_directory,
    create_file_names_list,
    compare_file_lists,
    get_images_from_list,
    extract_index,
)
from utils import mask_on_image, dilate_mask, plot_grid
from inpainting_iopaint_lama_sdxl import (
    inpaint_instance_iopaint_lama_sdxl,
    setup_diffusion_model,
    inpaint_batch_iopaint_lama_sdxl,
)
from inpainting_iopaint_lama import inpaint_batch_iopaint_lama
from inpainting_sdxl import inpaint_batch_sdxl

parser = argparse.ArgumentParser(description="Inpainting with iopaint")

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

parser.add_argument(
    "-e",
    "--end",
    default=np.Inf,
    type=int,
    help="Image index to start from",
)

parser.add_argument(
    "-d",
    "--device",
    default=2,
    type=int,
    help="Device to run the inpainting on",
)


args = parser.parse_args()


# set the device
# extract number from device string
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

# Set the environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Load the configuration file
config = toml.load(args.config)

# get id that is between "_" and "." in the config file name
id_tag = args.config.split("/")[-1].split("_", 1)[1].split(".")[0]

# Load YAML file into a dictionary
dataset_name = config["dataset"]["name"]
dataset_yaml_file = config["dataset"]["yaml_file"]
with open(dataset_yaml_file, "r") as file:
    datasets = yaml.safe_load(file)

# get dataset
dataset = datasets[dataset_name]


# get path with segmentation masks
# check which segmentation results should be used
if "use_existing" in config["segmentation"]:
    segmentation_id = config["segmentation"]["use_existing"]
    if segmentation_id == "gt":
        segmentation_masks_path = dataset["gt_seg_path"]
    else:
        segmentation_masks_path = f"/workspace/home/pipeline_outputs/{config['dataset']['name']}/{segmentation_id}/seg"
else:
    segmentation_masks_path = (
        f"/workspace/home/pipeline_outputs/{config['dataset']['name']}/{id_tag}/seg"
    )


# define output directories
output_dir = f"/workspace/home/pipeline_outputs/{config['dataset']['name']}/{id_tag}/inpainting_intermediate"

output_dir2 = (
    f"/workspace/home/pipeline_outputs/{config['dataset']['name']}/{id_tag}/inpainting"
)


# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)
    if args.start is None:
        segmentation_masks = get_images_from_directory(
            dir_path=segmentation_masks_path, ignore_copy=True
        )
    else:
        mask_names = create_file_names_list(segmentation_masks_path, ignore_copy=True)

else:
    print(
        "Output directory already exists. Check if there are still unsegmented images."
    )
    # check if all files are already segmented
    already_inpainted_images = create_file_names_list(output_dir2)
    to_be_inpainted_list = create_file_names_list(segmentation_masks_path)
    missing_files, _ = compare_file_lists(
        already_inpainted_images, to_be_inpainted_list
    )
    if len(missing_files) == 0:
        print("All images are already segmented.")
        sys.exit(0)
    else:
        print(f"Found {len(missing_files)} images to process.")
        # only process the missing files
        if args.start is None:
            segmentation_masks = get_images_from_list(
                missing_files, segmentation_masks_path
            )
        else:
            mask_names = missing_files

if args.start is not None:
    print(f"Starting from image index {args.start}")
    mask_names = [
        name
        for name in mask_names
        if extract_index(name) >= args.start and extract_index(name) < args.end
    ]
    segmentation_masks = get_images_from_list(
        mask_names, segmentation_masks_path, ignore_copy=True
    )


prompt = config["inpainting"]["prompt"]

# For dilating the mask
kernel_size = config["inpainting"]["dilation"]["kernel_size"]
iterations = config["inpainting"]["dilation"]["iterations"]
if (
    "width" in config["inpainting"]["dilation"]
    and "height" in config["inpainting"]["dilation"]
):
    size = (
        config["inpainting"]["dilation"]["width"],
        config["inpainting"]["dilation"]["height"],
    )
else:
    size = (299, 299)

Pipe, generator = setup_diffusion_model(args.device)

if "num_batches" in config["inpainting"]:
    num_batches = config["inpainting"]["num_batches"]
else:
    num_batches = 50


# seperate segmentation masks into batches
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for split in np.array_split(lst, n):
        yield split


for batch_index, mask_dicts in enumerate(
    tqdm(chunks(segmentation_masks, num_batches), total=num_batches)
):
    if "model" in config["inpainting"]:
        model_name = config["inpainting"]["model"]
    else:
        model_name = None  # default to lama and sdxl
    if model_name == "lama":
        # remove output_dir
        inpaint_batch_iopaint_lama(
            mask_dicts,
            dataset["raw_path"],
            kernel_size=kernel_size,
            iterations=iterations,
            size=size,
            save_dir=output_dir2,
        )
    elif model_name == "sdxl":
        inpaint_batch_sdxl(
            mask_dicts,
            dataset["raw_path"],
            prompt,
            Pipe,
            generator,
            kernel_size=kernel_size,
            iterations=iterations,
            size=size,
            save_dir=output_dir2,
        )
    else:
        inpaint_batch_iopaint_lama_sdxl(
            mask_dicts,
            dataset["raw_path"],
            prompt,
            Pipe,
            generator,
            kernel_size=kernel_size,
            iterations=iterations,
            size=size,
            save_dir=output_dir,
            save_dir2=output_dir2,
        )
    print(f"Finished batch {batch_index+1} of {num_batches}")

# for i, mask_dict in enumerate(tqdm(segmentation_masks)):
#    file_name = mask_dict["file_name"]
#
# pipe, generator = setup_diffusion_model(args.device)

# save_path2 = f"{output_dir2}/{file_name}"

# inpaint_instance_iopaint_lama_sdxl(
#     mask_dict,
#     dataset["raw_path"],
#     prompt,
#     pipe,
#     generator,
#     kernel_size=kernel_size,
#     iterations=iterations,
#     save_dir=output_dir,
#     save_path2=save_path2,
# )
