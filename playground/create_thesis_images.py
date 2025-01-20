from PIL import Image
import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import dilate_mask, mask_on_image

img_ids = [25, 257]
# [589, 607, 601, 609, 611, 619, 626, 631, 696, 702, 794, 952, 1025]
# [592,610,932,9040,600]
# [600]  # [607, 895, 641, 738, 779, 758, 782, 3421, 660, 664]
id_tags = ["random_inpaintings"]
tasks = ["masks"]


def convert_img_id(img_id):
    # 55292 -> ISIC_0055292.jpg
    # 10 -> ISIC_0000010.jpg
    return f"ISIC_{str(img_id).zfill(7)}.jpg"


# path to output directory
raw_dir = "/workspace/datasets/ISIC19/raw"
output_dir = "/workspace/home/pipeline_outputs/ISIC2019"
save_dir = f"{output_dir}/thesis_images"
os.makedirs(save_dir, exist_ok=True)


# get segmentation images and inpainting images
if False:
    for id_tag in id_tags:
        for task in tasks:
            for img_id in img_ids:
                img_id = convert_img_id(img_id)
                img_path = f"{output_dir}/{id_tag}/{task}/{img_id}"
                save_path = f"{save_dir}/{task}_{id_tag}_{img_id}"
                img = Image.open(img_path).resize((400, 400))
                if task == "seg":
                    # ensure the image is binary (black and white)
                    img = img.convert("L").point(lambda x: 255 if x > 128 else 0)
                img.save(save_path)

        print(f"Saved images for {id_tag}")

# get dilated segmentation and mask over image
if False:
    for id_tag in id_tags:
        for img_id in img_ids:
            img_id = convert_img_id(img_id)
            img_path = f"{raw_dir}/{img_id}"
            mask_path = f"{output_dir}/{id_tag}/seg/{img_id}"
            img = Image.open(img_path)
            mask = (
                Image.open(mask_path)
                .convert("L")
                .resize(img.size)
                .point(lambda x: 255 if x > 128 else 0)
            )
            dilated_mask = dilate_mask(mask, kernel_size=15, iterations=10)
            mask_over_img = mask_on_image(img, mask)
            dilated_mask_over_img = mask_on_image(img, dilated_mask)
            img.save(f"{save_dir}/img_{img_id}")
            mask.save(f"{save_dir}/mask_{id_tag}_{img_id}")
            dilated_mask.save(f"{save_dir}/dilated_mask_{id_tag}_{img_id}")
            mask_over_img.save(f"{save_dir}/mask_over_img_{id_tag}_{img_id}")
            dilated_mask_over_img.save(
                f"{save_dir}/dilated_mask_over_img_{id_tag}_{img_id}"
            )

        print(f"Saved images for {id_tag}")


if True:
    for id_tag in id_tags:
        for task in tasks:
            for img_id in img_ids:
                img_id = convert_img_id(img_id)
                img_path = f"{output_dir}/{id_tag}/{task}/{img_id}"
                save_path = f"{save_dir}/{task}_{id_tag}_{img_id}"
                img = Image.open(img_path).resize((400, 400))
                img.save(save_path)

        print(f"Saved images for {id_tag}")
