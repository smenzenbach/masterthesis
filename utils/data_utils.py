import re
from llama_index.core import SimpleDirectoryReader
import os
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt
import warnings
import csv

file_dir = os.getcwd()
rrr_root = os.path.abspath(os.path.join(file_dir, "/workspace/home/rrr"))

# Add the project root directory to sys.path
if rrr_root not in sys.path:
    sys.path.append(rrr_root)

import rrr.decoy_mnist as decoy_mnist


def load_images(
    dir_path="/home/data/ISIC2019/raw",
    confounder_dir=None,
    no_confounder_dir=None,
    ignore_copy=False,
):
    """
    If confounder_dir and no_confounder_dir are None, returns a list images with metadata for each image.
    If confounder_dir and no_confounder_dir are not None, returns two lists (images_confounder, images_no_confounder) with metadata for each image.


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

    seperate_dicts = confounder_dir is not None and no_confounder_dir is not None

    if seperate_dicts:
        images_confounder = []
        images_no_confounder = []

        for image in image_documents:
            metadata = metadata_str2dict(image.get_metadata_str())
            image_path = metadata["file_path"]

            if ignore_copy and "copy" in image_path.lower():
                continue

            folder_name = os.path.basename(os.path.dirname(image_path))

            if folder_name == confounder_dir:
                metadata[confounder_dir] = True
                # TODO: convert image to PIL?
                images_confounder.append({"image": image, "meta": metadata})

            elif folder_name == no_confounder_dir:
                metadata[confounder_dir] = False
                images_no_confounder.append({"image": image, "meta": metadata})

            else:
                print(
                    f"The jpg is not in neither the {confounder_dir} nor {no_confounder_dir} folder."
                )

        return images_confounder, images_no_confounder

    else:
        images = []

        for image in image_documents:
            metadata = metadata_str2dict(image.get_metadata_str())
            image_path = metadata["file_path"]
            if ignore_copy and "copy" in image_path.lower():
                continue
            images.append({"image": image, "meta": metadata})

        return images


def load_images_and_combine(
    dir_path="/home/data/ISIC2019/raw",
    confounder_dir="patch",
    no_confounder_dir="no_patch",
    num_imgs=None,
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

    # combine images with and without patch
    images = []
    if num_imgs is None:
        images.extend(images_confounder)
        images.extend(images_no_confounder)
    else:
        # get half of the images with patch and half of the images without patch
        # if not enough images, return all
        num_imgs_per_class = int(np.ceil(num_imgs / 2))
        num_imgs_confounder = num_imgs_per_class
        num_imgs_no_confounder = num_imgs_per_class

        if num_imgs_per_class > len(images_confounder):
            warnings.warn(
                f"Only {len(images_confounder)} images with confounder instead of {num_imgs_per_class} available. Returning all."
            )
            num_imgs_confounder = len(images_confounder)

        if num_imgs_per_class > len(images_no_confounder):
            warnings.warn(
                f"Only {len(images_no_confounder)} images without confounder instead of {num_imgs_per_class} available. Returning all."
            )
            num_imgs_no_confounder = len(images_no_confounder)

        images.extend(images_confounder[:num_imgs_confounder])
        images.extend(images_no_confounder[:num_imgs_no_confounder])

    return images


def save_mnist_images():
    # Xr,Xtr --> orginial MNIST; X,Xt --> decoy MNIST; E,Et --> explanations
    # y,yt --> same for decoy and orginial MNIST

    Xr, X, y, E, Xtr, Xt, yt, Et = decoy_mnist.generate_dataset(
        cachefile="../data/decoy-mnist.npz"
    )
    Xr = np.array([x.ravel() for x in Xr])
    Xtr = np.array([x.ravel() for x in Xtr])

    n_samples = 60000
    X = X.reshape((n_samples, 1, 28, 28))
    Xt = Xt.reshape((10000, 1, 28, 28))
    E = E.reshape(n_samples, 1, 28, 28)
    Xr = Xr.reshape((n_samples, 1, 28, 28))
    Xtr = Xtr.reshape((10000, 1, 28, 28))
    Et = Et.reshape(10000, 1, 28, 28)

    def save_images(X, output_dir, n_images=None):
        if n_images is None:
            n_images = X.shape[0]
        for i in range(n_images):
            print(f"Saving image {i}")
            plt.figure(figsize=(1, 1))
            plt.imshow(X[i, 0], cmap="gray")
            plt.axis("off")
            plt.savefig(
                os.path.join(output_dir, f"mnist_{i}.jpeg"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

    output_dir_confounder = "/workspace/home/data/DecoyMNIST/confounder"
    os.makedirs(output_dir_confounder, exist_ok=True)

    output_dir_no_confounder = "/workspace/home/data/DecoyMNIST/no_confounder"
    os.makedirs(output_dir_no_confounder, exist_ok=True)

    save_images(X, output_dir_confounder, 1000)
    save_images(Xr, output_dir_no_confounder, 1000)


# save_mnist_images()


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


def get_images_from_list(
    image_name_list, dir_path, sub_dir=False, without_img=False, ignore_copy=False
):
    def process_image(image_name, image_path):
        try:
            meta_dict = {"file_path": image_path, "file_name": image_name}
            if without_img:
                return meta_dict
            img_dict = {}
            image_path = os.path.join(image_name, image_path)
            image = Image.open(image_path)
            img_dict["image"] = image
            meta_dict = {"file_path": image_path, "file_name": image_name}
            img_dict["meta"] = meta_dict
            return img_dict
        except Exception as e:
            print(f"Error loading {image_name}: {e}")

    images = []
    # Traverse the folder and subfolders (if enabled)
    for root, _, files in os.walk(dir_path):
        if image_name_list is None:
            files = sorted(files)
            for file_name in files:
                if ignore_copy and "copy" in file_name.lower():
                    continue
                file_path = os.path.join(root, file_name)
                images.append(process_image(file_name, file_path))
        else:
            image_name_list = sorted(image_name_list)
            for file_name in image_name_list:
                if file_name in files:
                    if ignore_copy and "copy" in file_name.lower():
                        continue
                    file_path = os.path.join(root, file_name)
                    images.append(process_image(file_name, file_path))

        if not sub_dir:
            # Break after processing the top-level folder if not searching subfolders
            break

    return images


def get_images_from_directory(dir_path, sub_dir=False, without_img=False, ignore_copy=False):
    return get_images_from_list(None, dir_path, sub_dir, without_img, ignore_copy)


def find_next_id(folder_path):
    # Initialize the highest ID as 0
    highest_id = 0

    # Regular expression to find 4-digit numbers in filenames
    id_pattern = re.compile(r"\b(\d{4})\b")

    # Flag to check if any ID was found
    id_found = False

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Search for a 4-digit ID in the filename
        match = id_pattern.search(filename)
        if match:
            # Convert the found ID to an integer
            current_id = int(match.group(1))
            # Update the highest ID found
            highest_id = max(highest_id, current_id)
            id_found = True

    # Increment the highest ID by 1 for the next ID
    next_id = highest_id + 1 if id_found else 1
    # Ensure the ID remains 4 digits long (leading zeros if needed)
    next_id_str = f"{next_id:04}"

    return next_id_str


def sort_csv_by_column(csv_path, column_index, header=True):
    """
    Sort a CSV file by the values in a specific column. The sorted CSV file will overwrite the original file.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file to be sorted.
    column_index : int
        The index of the column to sort by.
    header : bool, optional
        Whether the CSV file contains a header row with column names, by default True
    """

    # Read the CSV file into a list of rows
    with open(csv_path, "r") as file:
        rows = list(csv.reader(file))

    # Sort the rows based on the values in the specified column
    sorted_rows = sorted(rows[1:], key=lambda row: row[column_index])

    # Add the header row back to the sorted list if it was present
    if header:
        sorted_rows.insert(0, rows[0])

    # Write the sorted rows back to the original CSV file
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(sorted_rows)


def extract_index(filename):
    """
    Extract the index from a filename in the format "image_index.jpg". Leading zeros are removed.

    Parameters
    ----------
    filename : str
        The filename to extract the index from.

    Returns
    -------
    int
        The extracted index as an integer.

    Raises
    ------
    Warning
        If the index is not found in the filename. In this case, the index 0 is returned.

    """
    index = filename.split("_")[1]
    # remove the file extension
    index = index.split(".")[0]
    # remove the leading zeros
    index = index.lstrip("0")
    if index == "":
        index = "0"
        warnings.warn(f"Index not found in filename {filename}. Using index 0.")
    index = int(index)
    return index
