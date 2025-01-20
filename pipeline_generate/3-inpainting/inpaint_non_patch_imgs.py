from diffusers import AutoPipelineForInpainting
import torch
import os, sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
import tempfile
import cv2
import subprocess
import toml

import os, sys
from tqdm import tqdm
import yaml
import toml
import argparse

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

from inpainting_iopaint_lama_sdxl import (
    inpaint_instance_iopaint_lama_sdxl,
    setup_diffusion_model,
    inpaint_batch_iopaint_lama_sdxl,
)
from inpainting_iopaint_lama import inpaint_batch_iopaint_lama
from inpainting_sdxl import inpaint_batch_sdxl


def setup_diffusion_model(device):

    device2 = f"cuda:{device}"
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to(device2)
    pipe.enable_attention_slicing()  # Recommended if your computer has < 64 GB of RAM

    generator = torch.Generator(device2).manual_seed(0)
    return pipe, generator


def create_random_mask(image_size):
    mask = np.zeros((image_size[1], image_size[0]))
    # create ellipse mask with higher probability at the edge of the image
    if np.random.rand() > 0.5:
        if np.random.rand() > 0.5:
            center_x = min(max(int(np.random.geometric(0.5)), 0), image_size[0] - 1)
        else:
            center_x = image_size[0] - min(
                max(int(np.random.geometric(0.5)), 0), image_size[0] - 1
            )
        center_y = np.random.randint(0, image_size[1])
        center = (center_x, center_y)
        angle = np.random.randint(80, 100)
    else:
        if np.random.rand() > 0.5:
            center_y = min(max(int(np.random.geometric(0.5)), 0), image_size[1] - 1)
        else:
            center_y = image_size[1] - min(
                max(int(np.random.geometric(0.5)), 0), image_size[1] - 1
            )
        center_x = np.random.randint(0, image_size[0])
        center = (center_x, center_y)
        angle = np.random.randint(-10, 10)
    axes = (
        np.random.randint(image_size[0] // 4, image_size[0] // 3),
        np.random.randint(image_size[1] // 4, image_size[1] // 3),
    )
    mask = cv2.ellipse(mask, center, axes, angle, 0, 360, (255, 255, 255), -1)
    mask = Image.fromarray(mask).convert("L")
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpainting with iopaint")

    parser.add_argument(
        "--model", type=str, required=True, help="Model to use for inpainting"
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

    dataset_yaml_file = "/workspace/home/data/datasets.yaml"
    with open(dataset_yaml_file, "r") as file:
        datasets = yaml.safe_load(file)
    dataset = datasets["ISIC2019"]

    # load images
    segmentation_masks_path = dataset["gt_seg_path"]
    segmentation_masks = create_file_names_list(segmentation_masks_path)
    all_images = create_file_names_list(dataset["raw_path"])
    # get images that are not yet segmented
    images_list = compare_file_lists(segmentation_masks, all_images)[0]

    if args.start is None:
        images = get_images_from_list(images_list, dataset["raw_path"])
    else:
        print(f"Starting from image index {args.start}")
        images_list = [
            name for name in images_list if extract_index(name) >= args.start
        ]
        images = get_images_from_list(images_list, dataset["raw_path"])

    # create output directories
    id_tag = f"random_inpaintings_{args.model}"
    output_dir = (
        f"/workspace/home/pipeline_outputs/ISIC2019/{id_tag}/inpainting_intermediate"
    )
    output_dir2 = f"/workspace/home/pipeline_outputs/ISIC2019/{id_tag}/inpainting"
    mask_dir = f"/workspace/home/pipeline_outputs/ISIC2019/{id_tag}/masks"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # set prompt
    prompt = "part of skin in homogenous skin color"

    kernel_size = 0
    iterations = 0
    Pipe, generator = setup_diffusion_model(args.device)

    num_batches = 50

    # seperate segmentation masks into batches
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for split in np.array_split(lst, n):
            yield split

    for batch_index, img_dicts in enumerate(
        tqdm(chunks(images, num_batches), total=num_batches)
    ):
        model_name = args.model

        mask_dicts = []
        for img_dict in img_dicts:
            img = Image.open(img_dict["file_path"])
            mask = create_random_mask(img.size)
            mask_path = os.path.join(mask_dir, img_dict["file_name"])
            mask.save(mask_path)
            mask_dicts.append(
                {"file_path": mask_path, "file_name": img_dict["file_name"]}
            )

        if model_name == "lama":
            # remove output_dir
            inpaint_batch_iopaint_lama(
                mask_dicts,
                dataset["raw_path"],
                kernel_size=kernel_size,
                iterations=iterations,
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
                save_dir=output_dir,
                save_dir2=output_dir2,
            )
        print(f"Finished batch {batch_index+1} of {num_batches}")
