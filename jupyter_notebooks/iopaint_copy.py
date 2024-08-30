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


# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils import load_images_and_combine, load_images
from utils import mask_on_image, dilate_mask, plot_grid

import argparse


# +
# __import__("pdb").set_trace()
parser = argparse.ArgumentParser(description="Inpainting with iopaint")
parser.add_argument(
    "-s",
    "--start",
    default="0",
    type=int,
    help="Number of images to skip before starting inpainting",
)

parser.add_argument(
    "-d",
    "--device",
    default="2",
    type=str,
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


# Load the YAML file into a dictionary
dataset_yaml_file = "/workspace/home/data/datasets.yaml"
with open(dataset_yaml_file, "r") as file:
    datasets = yaml.safe_load(file)

# Load images
dataset_key = "ISIC2019"  # 'Pascal_VOC07' #
dataset = datasets[dataset_key]
# images_confounder, images_no_confounder = load_images(dir_path=f"/workspace/home/data/{dataset['dataset_dir']}/raw", confounder_dir=dataset['confounder_dir'], no_confounder_dir=dataset['no_confounder_dir'])
# segmentation_masks_confounder, _ = load_images(dir_path=f"/workspace/home/data/{dataset['dataset_dir']}/seg", confounder_dir=dataset['confounder_dir'], no_confounder_dir=dataset['no_confounder_dir'])

segmentation_masks_confounder = load_images(
    dir_path=f"/workspace/datasets/ISIC19/segmentation", ignore_copy=True
)


output_dir = f"/workspace/home/outputfiles/{dataset['dataset_dir']}/inpainting/iopaint/{init_model}"

output_dir_confounder = os.path.join(output_dir, dataset["confounder_dir"])

output_dir_confounder2 = f"/workspace/home/data/{dataset['dataset_dir']}/inpainted/"
# output_dir_no_confounder = os.path.join(output_dir, dataset['no_confounder_dir'])

# Create output directories if they don't exist
os.makedirs(output_dir_confounder, exist_ok=True)
# os.makedirs(output_dir_no_confounder, exist_ok=True)


prompt = "part of skin in homogenous skin color"
# "part of skin"
# "remove the confounder"
# ""
# "remove colorful patch"
# "lesion on skin"


# Set the environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# for i, (image_dict, mask_dict) in enumerate(tqdm(zip(images_confounder, segmentation_masks_confounder))):
for i, mask_dict in enumerate(tqdm(segmentation_masks_confounder)):
    mask = mask_dict["image"]
    meta_mask = mask_dict["meta"]
    mask_url = meta_mask["file_path"]
    file_name = meta_mask["file_name"]

    index = file_name.split("_")[1]
    # remove the file extension
    index = index.split(".")[0]
    # remove the leading zeros
    index = index.lstrip("0")
    index = int(index)

    # skip images before the start index
    if index < args.start:
        continue

    # skip already inpainted images
    if os.path.exists(
        os.path.join(
            output_dir_confounder2,
            file_name,
        )
    ):
        filepath = os.path.join(output_dir_confounder2, file_name)
        print(f"\033[31m Already inpainted {filepath} \033[0m")
        continue

    img_url = mask_url.replace("segmentation", "raw")

    # Load the image
    image = Image.open(img_url).convert("RGB")
    original_img_size = image.size

    # Dilate the mask
    dilated_mask = dilate_mask(Image.open(mask_url), kernel_size=5, iterations=5)
    resized_img = image.resize(dilated_mask.shape[:2])

    with tempfile.TemporaryDirectory() as tmpdirname:

        # Define the path for the temporary mask file
        mask_path = os.path.join(tmpdirname, "mask.jpg")
        img_url = os.path.join(tmpdirname, file_name)
        # Save the dilated mask to the temporary file
        cv2.imwrite(mask_path, dilated_mask)
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
            f"--output={output_dir_confounder}",
        ]

        # Execute the command
        subprocess.run(command)

        # path for second iteration
        img_url = os.path.join(output_dir_confounder, file_name.replace(".jpg", ".png"))

        #!iopaint run --model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" --device={device} --image={img_url} --mask={mask_path} --output={output_dir_confounder2}

        image = Image.open(img_url)
        mask = Image.fromarray(dilated_mask)
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
        result.save(f"{output_dir_confounder2}/{file_name}")

        print(f"\033[31m Saved {output_dir_confounder2}/{file_name} \033[0m")


# for image_dict in images_no_confounder:
#   image = image_dict['image']
#   meta_img = image_dict['meta']
#   img_url = meta_img['file_path']

#   # save image as PIL image
#   img = Image.open(img_url)
#   img.save(f"{output_dir_no_confounder}/{meta_img['file_name']}")
