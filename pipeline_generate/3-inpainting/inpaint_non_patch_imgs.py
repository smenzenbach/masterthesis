from diffusers import AutoPipelineForInpainting
from networkx import center
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
    mask = Image.fromarray(mask)
    return mask


def inpaint_instance_iopaint_lama_sdxl(
    images_dict,
    prompt,
    pipeline,
    generator,
    save_mask_dir=None,
    save_dir=None,
    save_path2=None,
):
    img_path = images_dict["file_path"]
    file_name = images_dict["file_name"]

    # check if already inpainted
    if save_path2 is not None and os.path.exists(save_path2):
        print(f"Already inpainted {save_path2}")
        return

    img = Image.open(img_path).convert("RGB")

    mask = create_random_mask(img.size)

    # save mask
    mask.convert("RGB").save(f"{save_mask_dir}/{file_name}")

    original_img_size = img.size

    # resized_img = image.resize(dilated_mask.shape[:2])

    with tempfile.TemporaryDirectory() as tmpdirname:

        # Define the path for the temporary mask file
        mask_temp_path = os.path.join(tmpdirname, "mask.jpg")
        img_temp_path = os.path.join(tmpdirname, file_name)
        if save_dir:
            output_dir = save_dir
        else:
            output_dir = os.path.join(tmpdirname, "output")

        # Save the mask to the temporary file
        cv2.imwrite(mask_temp_path, np.array(mask))

        init_model = "lama"  # @param ['lama', 'sd1.5', 'paint_by_example', 'ldm']
        device = "cuda"  # @param ['cuda', 'cpu', 'mps']

        # Construct the command as a list of arguments
        command = [
            "iopaint",
            "run",
            f"--model={init_model}",
            f"--device={device}",
            f"--image={img_path}",
            f"--mask={mask_temp_path}",
            f"--output={output_dir}",
        ]

        # Execute the command
        subprocess.run(command)

        # path for second iteration
        inpainted_path = os.path.join(output_dir, file_name.replace(".jpg", ".png"))

        #!iopaint run --model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" --device={device} --image={img_url} --mask={mask_path} --output={output_dir_confounder2}

        image = Image.open(inpainted_path)
        image = image.resize((1024, 1024))
        mask = mask.resize((1024, 1024))
        # Run the diffuser
        result2 = pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=8.0,
            num_inference_steps=25,  # steps between 15 and 30 work well for us
            strength=0.99,  # make sure to use `strength` below 1.0
            generator=generator,
            # padding_mask_crop=32,
        ).images[0]

        result2 = result2.resize(original_img_size)

        if save_path2 is not None:
            result2.save(save_path2)
            print(f"\033[31m Saved {save_path2} \033[0m")
        else:
            result1 = Image.open(inpainted_path)
            return result1, result2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpainting with iopaint")

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
    id_tag = "random_inpaintings"
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

    # inpaint images
    for i, img_dict in enumerate(tqdm(images)):
        file_name = img_dict["file_name"]

        save_path2 = f"{output_dir2}/{file_name}"

        if save_path2 is not None and os.path.exists(save_path2):
            print(f"Already inpainted {save_path2}")
            continue

        pipe, generator = setup_diffusion_model(args.device)

        inpaint_instance_iopaint_lama_sdxl(
            img_dict,
            prompt,
            pipe,
            generator,
            save_mask_dir=mask_dir,
            save_dir=output_dir,
            save_path2=save_path2,
        )
