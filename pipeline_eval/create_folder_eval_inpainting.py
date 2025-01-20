import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import get_images_from_directory

id_tags = [
    "random_inpaintings_sdxl",
]  # ["random_inpaintings", "gt"]

seg_folder = "/workspace/datasets/ISIC19/segmentation"
patch_images = sorted(os.listdir(seg_folder))

output_dir = "/workspace/home/pipeline_outputs/ISIC2019"

for id_tag in id_tags:
    processed_path = os.path.join(output_dir, id_tag, "h5_files/processed")
    cancer_imgs_path = os.path.join(processed_path, "cancer")
    non_cancer_imgs_path = os.path.join(processed_path, "no_cancer")
    cancer_imgs = get_images_from_directory(cancer_imgs_path)
    non_cancer_imgs = get_images_from_directory(non_cancer_imgs_path)

    patch_dir = os.path.join(output_dir, id_tag, "patch")
    os.makedirs(patch_dir, exist_ok=True)
    no_patch_dir = os.path.join(output_dir, id_tag, "no_patch")
    os.makedirs(no_patch_dir, exist_ok=True)

    for img_dict in non_cancer_imgs:
        img_name = img_dict["file_name"]
        img_path = img_dict["file_path"]
        # create softlinks from inpainted images to patch and no_patch folders
        if img_name in patch_images:
            os.symlink(img_path, os.path.join(patch_dir, img_name))
        else:
            os.symlink(img_path, os.path.join(no_patch_dir, img_name))
    for img_dict in cancer_imgs:
        img_name = img_dict["file_name"]
        img_path = img_dict["file_path"]
        # create softlinks from inpainted images to patch and no_patch folders
        os.symlink(img_path, os.path.join(no_patch_dir, img_name))


# for raw images
if False:
    processed_path = "/workspace/datasets/ISIC19/processed"
    cancer_imgs_path = os.path.join(processed_path, "cancer")
    non_cancer_imgs_path = os.path.join(processed_path, "no_cancer")
    cancer_imgs = get_images_from_directory(cancer_imgs_path)
    non_cancer_imgs = get_images_from_directory(non_cancer_imgs_path)

    skipped_imgs = [
        "ISIC_0024260.jpg",
        "ISIC_0024269.jpg",
        "ISIC_0024281.jpg",
        "ISIC_0036065.jpg",
    ]
    if os.path.exists("/workspace/home/data/ISIC2019/h5_files/skipped_images.txt"):
        # load skipped images
        with open(
            "/workspace/home/data/ISIC2019/h5_files/skipped_images.txt", "r"
        ) as f:
            doubles_skipped_imgs = f.readlines()
            doubles_skipped_imgs = [x.strip() for x in doubles_skipped_imgs]
            skipped_imgs.extend(doubles_skipped_imgs)

    raw_output_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_output_dir, exist_ok=True)
    patch_dir = os.path.join(raw_output_dir, "patch")
    os.makedirs(patch_dir, exist_ok=True)
    no_patch_dir = os.path.join(raw_output_dir, "no_patch")
    os.makedirs(no_patch_dir, exist_ok=True)

    for img_dict in non_cancer_imgs:
        img_name = img_dict["file_name"]
        img_path = img_dict["file_path"]
        # create softlinks from inpainted images to patch and no_patch folders
        if img_name in patch_images and img_name not in skipped_imgs:
            os.symlink(img_path, os.path.join(patch_dir, img_name))
        elif img_name not in skipped_imgs:
            os.symlink(img_path, os.path.join(no_patch_dir, img_name))
    for img_dict in cancer_imgs:
        img_name = img_dict["file_name"]
        img_path = img_dict["file_path"]
        # create softlinks from inpainted images to patch and no_patch folders
        if img_name not in skipped_imgs:
            os.symlink(img_path, os.path.join(no_patch_dir, img_name))
