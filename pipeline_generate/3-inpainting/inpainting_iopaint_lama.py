from re import I
from diffusers import AutoPipelineForInpainting
from sympy import im
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


# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from utils import dilate_mask


def inpaint_batch_iopaint_lama(
    mask_dict_list,
    raw_images_path,
    kernel_size=5,
    iterations=10,
    size=(299, 299),
    save_dir=None,
):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Define the path for the temporary mask file
        masks_temp_path = os.path.join(tmpdirname, f"masks")
        imgs_temp_path = os.path.join(tmpdirname, f"images")

        # Create the directories
        os.makedirs(masks_temp_path, exist_ok=True)
        os.makedirs(imgs_temp_path, exist_ok=True)

        # Save the dilated mask and images to the temporary file
        for mask_dict in tqdm(mask_dict_list):
            mask_path = mask_dict["file_path"]
            file_name = mask_dict["file_name"]

            # check if already inpainted
            final_inpainting_file = os.path.join(
                save_dir, file_name.replace(".jpg", ".png")
            )
            img_temp_path = os.path.join(imgs_temp_path, file_name)
            mask_temp_path = os.path.join(masks_temp_path, file_name)
            if final_inpainting_file is not None and os.path.exists(
                final_inpainting_file
            ):
                print(f"Already inpainted {final_inpainting_file}")
                continue

            img = Image.open(f"{raw_images_path}/{file_name}").convert("RGB")

            mask = (
                Image.open(mask_path)
                .convert("L")
                .resize(img.size)
                .point(lambda x: 255 if x > 128 else 0)
            )
            dilated_mask = dilate_mask(
                mask,
                kernel_size=kernel_size,
                iterations=iterations,
            )
            dilated_mask = dilated_mask.resize(img.size)

            cv2.imwrite(img_temp_path, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_temp_path, np.array(dilated_mask))

        init_model = "lama"  # @param ['lama', 'sd1.5', 'paint_by_example', 'ldm']
        device = "cuda"  # @param ['cuda', 'cpu', 'mps']

        # Construct the command as a list of arguments
        command = [
            "iopaint",
            "run",
            f"--model={init_model}",
            f"--device={device}",
            f"--image={imgs_temp_path}",
            f"--mask={masks_temp_path}",
            f"--output={save_dir}",
        ]

        # Run the command
        subprocess.run(command)
