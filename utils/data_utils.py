import re
from llama_index.core import SimpleDirectoryReader
import os
import numpy as np
from PIL import Image
import warnings
import csv
import h5py


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


def file_name_from_file_path(filepath):
    """Extracts the file name from a given filepath"""
    # Split the path into components and return the last component
    return os.path.basename(filepath)


def create_file_names_list(dir_path, sub_dir=False, ignore_copy=False):
    file_list = []
    for _, _, files in os.walk(dir_path):
        file_list.extend(files)
        if not sub_dir:
            # Break after processing the top-level folder if not searching subfolders
            break
    if ignore_copy:
        file_list = [
            file_name for file_name in file_list if "copy" not in file_name.lower()
        ]
    return sorted(file_list)


def create_file_names_list_from_csv(
    csv_file_path, header=True, filter_func=None, ignore_copy=False, num_imgs=None
):
    with open(csv_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        if header:
            try:
                next(csv_reader)
            except StopIteration:
                print(f"Empty CSV file: {csv_file_path}")
                return []
        file_list = []
        counter = 0
        for row in csv_reader:
            if num_imgs is not None and counter >= num_imgs:
                print(f"{num_imgs} images added to list.")
                break
            if filter_func is None:
                if ignore_copy and "copy" in row[0].lower():
                    continue
                file_list.append(row[0])
                counter += 1
            else:
                if filter_func([int(x) if x.isdigit() else x for x in row]) and (
                    not ignore_copy or "copy" not in row[0].lower()
                ):
                    file_list.append(row[0])
                    counter += 1
        if num_imgs is not None and len(file_list) < num_imgs:
            print(
                f"Only {len(file_list)} images found in the CSV file, less than the requested {num_imgs}."
            )
    return file_list


def get_images_from_list(
    file_name_list, dir_path, sub_dir=False, as_dict=False, ignore_copy=True
):
    """Returns per default a list of file paths of the files in the given list. If as_dict is True, returns a list of dictionaries with the keys 'file_path', 'file_name' and 'imgage'."""

    def process_image(image_name, image_path):
        try:
            if not as_dict:
                return image_path
            else:
                image = Image.open(image_path)
                img_dict = {
                    "file_path": image_path,
                    "file_name": image_name,
                    "image": image,
                }
                return img_dict
        except Exception as e:
            print(f"Error loading {image_name}: {e}")

    images = []
    # Traverse the folder and subfolders (if enabled)
    for root, _, files in os.walk(dir_path):
        if file_name_list is None:
            files = sorted(files)
            for file_name in files:
                if ignore_copy and "copy" in file_name.lower():
                    continue
                file_path = os.path.join(root, file_name)
                images.append(process_image(file_name, file_path))
        else:
            file_name_list = sorted(file_name_list)
            for file_name in file_name_list:
                if file_name in files:
                    if ignore_copy and "copy" in file_name.lower():
                        continue
                    file_path = os.path.join(root, file_name)
                    images.append(process_image(file_name, file_path))

        if not sub_dir:
            # Break after processing the top-level folder if not searching subfolders
            break

    # sort images according to filename
    if as_dict:
        images = sorted(images, key=lambda x: x["file_name"])
    else:
        # Function to extract the numeric part from the filename
        def extract_number(path):
            return int(re.search(r"(\d+)\.jpg$", path).group(1))

        # Sorting based on the extracted number
        images = sorted(images, key=extract_number)
    return images


def get_images_from_directory(dir_path, sub_dir=False, as_dict=False, ignore_copy=True):
    return get_images_from_list(None, dir_path, sub_dir, as_dict, ignore_copy)


def get_images_from_csv(
    csv_file_path,
    dir_path,
    sub_dir=False,
    as_dict=False,
    ignore_copy=True,
    header=True,
    filter_func=None,
    num_imgs=None,
):
    image_names = create_file_names_list_from_csv(
        csv_file_path, header, filter_func, ignore_copy, num_imgs
    )
    if as_dict:
        print(f"{len(image_names)} image dicts added to list from {csv_file_path}.")
    else:
        print(f"{len(image_names)} image paths added to list from {csv_file_path}.")
    return get_images_from_list(image_names, dir_path, sub_dir, as_dict, ignore_copy)


def get_n_confounded_m_non_confounded_images(
    dir_path,
    gt_label_csv_path,
    n,
    m,
    sub_dir=False,
    as_dict=False,
    ignore_copy=True,
    label_column=1,
):
    # Load the ground truth labels from the CSV file
    with open(gt_label_csv_path, "r") as file:
        reader = csv.reader(file)
        # Skip the header row
        next(reader)
        gt_labels = {row[0]: row[label_column] for row in reader}

    # Get the list of image names from csv file (in case images are in subdirectories)
    image_names = list(gt_labels.keys())

    # Get the images from the directory
    images = get_images_from_list(image_names, dir_path, sub_dir, as_dict, ignore_copy)

    # Separate the images into confounded and non-confounded groups
    confounded_images = []
    non_confounded_images = []
    n_confounded = 0
    n_non_confounded = 0
    for image in images:
        if not as_dict:
            image_name = file_name_from_file_path(image)
        else:
            image_name = image["file_name"]
        if gt_labels[image_name] == "1" and n_confounded < n:
            confounded_images.append(image)
            n_confounded += 1
        elif gt_labels[image_name] == "0" and n_non_confounded < m:
            non_confounded_images.append(image)
            n_non_confounded += 1
        if n_confounded > n and n_non_confounded > m:
            break

    return confounded_images, non_confounded_images


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
    if header:
        sorted_rows = sorted(rows[1:], key=lambda row: row[column_index])
        # Add the header row back to the sorted list if it was present
        sorted_rows.insert(0, rows[0])
    else:
        sorted_rows = sorted(rows, key=lambda row: row[column_index])

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


def merge_csv_files(output_filename, input_filenames, sort=True, delete_merged=True):
    with open(output_filename, "w", newline="") as out_file:
        writer = csv.writer(out_file)
        header_written = False
        for filename in input_filenames:
            with open(filename, "r") as in_file:
                reader = csv.reader(in_file)
                header = next(reader)
                if not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)
            if delete_merged:
                os.remove(filename)
                print(f"Temporary file {filename} cleaned up.")
        if sort:
            sort_csv_by_column(output_filename, 0, header=True)


def read_h5(h5_file_path):
    with h5py.File(h5_file_path, "r") as f:
        return {key: f[key][()] for key in f.keys()}


def create_pascal_gt_csv(confounded_dir, raw_dir, csv_path):
    """Create a CSV file with ground truth labels for the Pascal VOC dataset. The first column is the image filename, and the second column is the ground truth label (0 or 1). The label is 1 for confounded images and 0 for non-confounded images."""
    confounded_images = create_file_names_list(confounded_dir)
    raw_images = create_file_names_list(raw_dir, sub_dir=True)
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_file_name", "label"])
        for image in raw_images:
            label = 1 if image in confounded_images else 0
            writer.writerow([image, label])
    sort_csv_by_column(csv_path, 0, header=True)
    print(f"Ground truth CSV file created at {csv_path}.")


def create_pascal_horses_gt_csv(confounded_dir, not_cofounded_dir, csv_path):
    confounded_images = create_file_names_list(confounded_dir)
    not_cofounded_dir = create_file_names_list(not_cofounded_dir)
    raw_images = confounded_images + not_cofounded_dir
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_file_name", "label"])
        for image in raw_images:
            label = 1 if image in confounded_images else 0
            writer.writerow([image, label])
    sort_csv_by_column(csv_path, 0, header=True)
    print(f"Ground truth CSV file created at {csv_path}.")
