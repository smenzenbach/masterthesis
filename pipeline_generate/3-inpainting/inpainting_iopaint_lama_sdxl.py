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


# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from utils import dilate_mask


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


def inpaint_instance_iopaint_lama_sdxl(
    mask_dict,
    raw_images_path,
    prompt,
    pipeline,
    generator,
    kernel_size=5,
    iterations=10,
    save_dir=None,
    save_path2=None,
):
    mask_path = mask_dict["file_path"]
    file_name = mask_dict["file_name"]

    # check if already inpainted
    if save_path2 is not None and os.path.exists(save_path2):
        print(f"Already inpainted {save_path2}")
        return

    mask = Image.open(mask_path).convert("L")

    img = Image.open(f"{raw_images_path}/{file_name}").convert("RGB")

    original_img_size = img.size

    dilated_mask = dilate_mask(
        Image.open(mask_path), kernel_size=kernel_size, iterations=iterations
    )
    resized_img = img.resize(dilated_mask.size)
    # resized_img = image.resize(dilated_mask.shape[:2])

    with tempfile.TemporaryDirectory() as tmpdirname:

        # Define the path for the temporary mask file
        mask_temp_path = os.path.join(tmpdirname, "mask.jpg")
        img_temp_path = os.path.join(tmpdirname, file_name)
        if save_dir:
            output_dir = save_dir
        else:
            output_dir = os.path.join(tmpdirname, "output")

        # Save the dilated mask to the temporary file
        cv2.imwrite(mask_temp_path, np.array(dilated_mask))
        resized_img_array = np.array(resized_img)
        cv2.imwrite(img_temp_path, cv2.cvtColor(resized_img_array, cv2.COLOR_RGB2BGR))

        init_model = "lama"  # @param ['lama', 'sd1.5', 'paint_by_example', 'ldm']
        device = "cuda"  # @param ['cuda', 'cpu', 'mps']

        # Construct the command as a list of arguments
        command = [
            "iopaint",
            "run",
            f"--model={init_model}",
            f"--device={device}",
            f"--image={img_temp_path}",
            f"--mask={mask_temp_path}",
            f"--output={output_dir}",
        ]

        # Execute the command
        subprocess.run(command)

        # path for second iteration
        inpainted_path = os.path.join(output_dir, file_name.replace(".jpg", ".png"))

        #!iopaint run --model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" --device={device} --image={img_url} --mask={mask_path} --output={output_dir_confounder2}

        image = Image.open(inpainted_path)
        mask = dilated_mask
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
