from operator import index
from diffusers import AutoPipelineForInpainting
import torch
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
import tempfile
import cv2
import subprocess
import toml


# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils import get_images_from_directory, extract_index
from utils import mask_on_image, dilate_mask, plot_grid

import argparse


# +
# __import__("pdb").set_trace()
parser = argparse.ArgumentParser(description="Inpainting with iopaint")

parser.add_argument(
    "--config", required=True, type=str, help="Path to the configuration file"
)

parser.add_argument(
    "-s",
    "--start",
    default=0,
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


init_model = "lama"  # @param ['lama', 'sd1.5', 'paint_by_example', 'ldm']
port = 4242
device = "cuda"  # @param ['cuda', 'cpu', 'mps']

cmd = (
    f"iopaint start --model {init_model} --host 0.0.0.0 --port {port} --device {device}"
)
print(cmd)


device2 = f"cuda:{args.device}"
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = pipe.to(device2)
pipe.enable_attention_slicing()  # Recommended if your computer has < 64 GB of RAM

generator = torch.Generator(device2).manual_seed(0)

# Load the configuration file
config = toml.load(args.config)

# get id that is between "_" and "." in the config file name
id_tag = int(args.config.split("_")[-1].split(".")[0])

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
        segmentation_masks_path = f"/workspace/home/pipeline_outputs/{config["dataset"]["name"]}/{segmentation_id}/seg"
else:
    segmentation_masks_path = f"/workspace/home/pipeline_outputs/{config["dataset"]["name"]}/{id_tag}/seg"


segmentation_masks = get_images_from_directory(dir_path=segmentation_masks_path, ignore_copy=True)


output_dir = f"/workspace/home/pipeline_outputs/{config["dataset"]["name"]}/{id_tag}/inpainting_intermediate"


output_dir2 = f"/workspace/home/pipeline_outputs/{config["dataset"]["name"]}/{id_tag}/inpainting"
# output_dir_no_confounder = os.path.join(output_dir, dataset['no_confounder_dir'])

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)

prompt = config["inpainting"]["prompt"]
# "part of skin"
# "remove the confounder"
# ""
# "remove colorful patch"
# "lesion on skin"


# Set the environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# for i, (image_dict, mask_dict) in enumerate(tqdm(zip(images_confounder, segmentation_masks_confounder))):
for i, mask_dict in enumerate(tqdm(segmentation_masks)):
    mask = mask_dict["image"]
    meta_mask = mask_dict["meta"]
    mask_url = meta_mask["file_path"]
    file_name = meta_mask["file_name"]

    index = extract_index(file_name)

    # skip images before the start index
    if index < args.start:
        continue

    # skip already inpainted images
    if os.path.exists(
        os.path.join(
            output_dir2,
            file_name,
        )
    ):
        filepath = os.path.join(output_dir2, file_name)
        print(f"\033[31m Already inpainted {filepath} \033[0m")
        continue

    img_url = f"{dataset['raw_path']}/{file_name}"

    # Load the image
    image = Image.open(img_url).convert("RGB")
    original_img_size = image.size

    # Dilate the mask
    kernel_size = config["inpainting.dilation"]["kernel_size"]
    iterations = config["inpainting.dilation"]["iterations"]
    dilated_mask = dilate_mask(Image.open(mask_url), kernel_size=kernel_size, iterations=iterations)
    resized_img = image.resize(dilated_mask.size)
    # resized_img = image.resize(dilated_mask.shape[:2])

    with tempfile.TemporaryDirectory() as tmpdirname:

        # Define the path for the temporary mask file
        mask_path = os.path.join(tmpdirname, "mask.jpg")
        img_url = os.path.join(tmpdirname, file_name)
        # Save the dilated mask to the temporary file
        cv2.imwrite(mask_path, np.array(dilated_mask))
        resized_img_array = np.array(resized_img)
        cv2.imwrite(img_url, cv2.cvtColor(resized_img_array, cv2.COLOR_RGB2BGR))

        # Construct the command as a list of arguments
        command = [
            "iopaint",
            "run",
            f"--model={init_model}",
            f"--device={device}",
            f"--image={img_url}",
            f"--mask={mask_path}",
            f"--output={output_dir}",
        ]

        # Execute the command
        subprocess.run(command)

        # path for second iteration
        img_url = os.path.join(output_dir, file_name.replace(".jpg", ".png"))

        #!iopaint run --model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" --device={device} --image={img_url} --mask={mask_path} --output={output_dir_confounder2}

        image = Image.open(img_url)
        mask = dilated_mask
        image = image.resize((1024, 1024))
        mask = mask.resize((1024, 1024))
        # Run the diffuser
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=8.0,
            num_inference_steps=25,  # steps between 15 and 30 work well for us
            strength=0.99,  # make sure to use `strength` below 1.0
            generator=generator,
            # padding_mask_crop=32,
        ).images[0]

        result = result.resize(original_img_size)

        # Save the inpainted image
        result.save(f"{output_dir2}/{file_name}")

        print(f"\033[31m Saved {output_dir2}/{file_name} \033[0m")
