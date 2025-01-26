from diffusers import FluxFillPipeline
import torch
import os, sys
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from utils import file_name_from_file_path, inpainting_preprocessing

login("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")


def setup_diffusion_model_flux(device):

    device_str = f"cuda:{device}"
    pipeline = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16,
    ).to(device_str)
    # pipeline.enable_attention_slicing()  # Recommended if your computer has < 64 GB of RAM
    # pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

    generator = torch.Generator(device_str).manual_seed(0)
    return pipeline, generator


def run_flux(image, mask, prompt, pipeline, generator):
    result_flux = pipeline(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=1632,
        width=1232,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=generator,
    ).images[0]
    return result_flux


def inpaint_instance_flux(
    mask_path,
    image_path,
    prompt,
    pipeline,
    generator,
    dilation_size=None,
    kernel_size=5,
    iterations=4,
    save_dir=None,
):
    # Save the dilated mask and images to the temporary file
    file_name = file_name_from_file_path(mask_path)

    # check if already inpainted
    if save_dir is not None and os.path.exists(save_dir):
        print(f"Already inpainted {save_dir}")
        return

    # get image and dilated mask
    img, dilated_mask = inpainting_preprocessing(
        image_path, mask_path, dilation_size, kernel_size, iterations
    )

    # run inpainting using sdxl
    result_flux = run_flux(
        img,
        dilated_mask,
        prompt,
        pipeline,
        generator,
    )

    # save or return final results
    if save_dir is not None:
        save_path_final = os.path.join(save_dir, file_name)
        result_flux.save(save_path_final)
        print(f"\033[31m Saved {save_path_final} \033[0m")
    else:
        return result_flux


def inpaint_batch_flux(
    mask_path_list,
    image_path_list,
    prompt,
    pipeline,
    generator,
    dilation_size=None,
    kernel_size=5,
    iterations=4,
    save_dir=None,
):
    # Save the dilated mask and images to the temporary file
    for mask_path, image_path in tqdm(zip(mask_path_list, image_path_list)):
        file_name = file_name_from_file_path(mask_path)

        # check if already inpainted
        final_inpainting_file = os.path.join(save_dir, file_name)
        if final_inpainting_file is not None and os.path.exists(final_inpainting_file):
            print(f"Already inpainted {final_inpainting_file}")
            continue

        # get image and dilated mask
        img, dilated_mask = inpainting_preprocessing(
            image_path, mask_path, dilation_size, kernel_size, iterations
        )

        # run inpainting using sdxl
        result_flux = run_flux(
            img,
            dilated_mask,
            prompt,
            pipeline,
            generator,
        )

        # save inpainted image
        result_flux.save(final_inpainting_file)
