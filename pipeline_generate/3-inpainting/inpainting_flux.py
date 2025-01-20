from diffusers import FluxFillPipeline
import torch
import os, sys
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from utils import dilate_mask

login("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")


def setup_diffusion_model_flux(device):

    device2 = f"cuda:{device}"
    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16
    ).to(device2)
    pipe.enable_attention_slicing()  # Recommended if your computer has < 64 GB of RAM
    # pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

    generator = torch.Generator(device2).manual_seed(0)
    return pipe, generator


def inpaint_batch_flux(
    mask_dict_list,
    raw_images_path,
    prompt,
    pipeline,
    generator,
    kernel_size=5,
    iterations=10,
    size=(299, 299),
    save_dir=None,
):
    # Save the dilated mask and images to the temporary file
    for mask_dict in tqdm(mask_dict_list):
        mask_path = mask_dict["file_path"]
        file_name = mask_dict["file_name"]

        # check if already inpainted
        final_inpainting_file = os.path.join(save_dir, file_name)
        if final_inpainting_file is not None and os.path.exists(final_inpainting_file):
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

        #!iopaint run --model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" --device={device} --image={img_url} --mask={mask_path} --output={output_dir_confounder2}

        image = img.resize((1024, 1024))
        mask = dilated_mask.resize((1024, 1024))
        # Run the diffuser
        result2 = pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=1632,
            width=1232,
            guidance_scale=30,
            num_inference_steps=50,
            max_sequence_length=512,
        ).images[0]

        result2 = result2.resize(img.size)

        result2.save(final_inpainting_file)


def inpaint_flux(
    mask_dict,
    raw_images_path,
    prompt,
    pipeline,
    generator,
    kernel_size=5,
    iterations=10,
    size=(299, 299),
):
    # Save the dilated mask and images to the temporary file
    mask_path = mask_dict["file_path"]
    file_name = mask_dict["file_name"]

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

    #!iopaint run --model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" --device={device} --image={img_url} --mask={mask_path} --output={output_dir_confounder2}

    image = img.resize((1024, 1024))
    mask = dilated_mask.resize((1024, 1024))
    # Run the diffuser
    result2 = pipeline(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=1632,
        width=1232,
        guidance_scale=15,
        num_inference_steps=25,
        # max_sequence_length=512,
    ).images[0]

    result2 = result2.resize(img.size)

    return result2
