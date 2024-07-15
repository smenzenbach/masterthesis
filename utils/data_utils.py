from calendar import c
from llama_index.core import SimpleDirectoryReader
import os
import numpy as np


def load_images(
    dir_path="/home/data/ISIC2019/raw",
    confounder_dir="patch",
    no_confounder_dir="no_patch",
):
    """
    Returns ISIC2019 images in two lists (images_patch, images_no_patch)

    Example metadata output:
    print(images_patch[0]['meta'])
    > {'file_path': '/home/data/ISIC2019/raw/patch/ISIC_0000557.jpg', 'file_name': 'ISIC_0000557.jpg', 'file_type': 'image/jpeg', 'file_size': '2630625', 'creation_date': '2024-05-20', 'last_modified_date': '2024-05-20', 'patch': True}
    """
    # load as image documents
    image_documents = SimpleDirectoryReader(
        dir_path,
        required_exts=[".jpeg", ".jpg", ".png"],
        recursive=True,
        filename_as_id=True,
    ).load_data()
    images_patch = []
    images_no_patch = []

    for image in image_documents:
        metadata = metadata_str2dict(image.get_metadata_str())
        image_path = metadata["file_path"]

        folder_name = os.path.basename(os.path.dirname(image_path))

        if folder_name == confounder_dir:
            metadata[confounder_dir] = True
            images_patch.append({"image": image, "meta": metadata})

        elif folder_name == no_confounder_dir:
            metadata[confounder_dir] = False
            images_no_patch.append({"image": image, "meta": metadata})

        else:
            print(
                f"The jpg is not in neither the {confounder_dir} nor {no_confounder_dir} folder."
            )

    return images_patch, images_no_patch


def load_images_and_combine(
    dir_path="/home/data/ISIC2019/raw",
    confounder_dir="patch",
    no_confounder_dir="no_patch",
    num_imgs=100,
):
    """
    Returns one list with ISIC2019 images half of images_patch and half of images_no_patch with a total of num_imgs

    Example metadata output:
    print(image_patch[0]['meta'])
    > {'file_path': '/home/data/ISIC2019/raw/patch/ISIC_0000557.jpg', 'file_name': 'ISIC_0000557.jpg', 'file_type': 'image/jpeg', 'file_size': '2630625', 'creation_date': '2024-05-20', 'last_modified_date': '2024-05-20', 'patch': True}
    """
    images_confounder, images_no_confounder = load_images(
        dir_path, confounder_dir, no_confounder_dir
    )

    num_imgs_per_class = int(np.ceil(num_imgs / 2))

    # combine images with and without patch
    images = []
    images.extend(images_confounder[:num_imgs_per_class])
    images.extend(images_no_confounder[:num_imgs_per_class])

    return images


def metadata_str2dict(string):
    # Split the string into lines
    lines = string.strip().split("\n")

    # Create a dictionary to store the key-value pairs
    file_info_dict = {}

    # Parse each line and add key-value pairs to the dictionary
    for line in lines:
        key, value = line.split(": ", 1)
        file_info_dict[key] = value

    return file_info_dict
