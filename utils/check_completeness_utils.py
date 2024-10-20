import os
import csv
import numpy as np


def compare_file_lists(files_currently_processed, files_to_be_processed):
    """Checking if all files in files_to_be_processed are in files_currently_processed, i.e., if all files that should be processed are processed.
    If files_to_be_processed contains copies, they are ignored.

    Parameters:
    -----------
    files_currently_processed: list
        List of files that are currently processed.
    files_to_be_processed: list
        List of files that should be processed.

    Returns:
    --------
    missing_files: list
        List of files that are in files_to_be_processed but not in files_currently_processed.
    copied_files: list
        List of files that are copies.
    """
    missing_files = []
    copied_files = []
    for file in files_to_be_processed:
        if "copy" in file.lower():
            copied_files.append(file)
            continue
        if file not in files_currently_processed:
            missing_files.append(file)
    # sort the list
    missing_files.sort()
    copied_files.sort()
    return missing_files, copied_files


def get_not_existent_indices_gt(gt_files):
    """
    Get the ground truth files indices that are not existing and break the incrementing sequence.
    """
    indices = file_name_to_index(gt_files)

    min_index = min(indices)
    max_index = max(indices)

    missing_indices = []
    for i in range(int(min_index), int(max_index) + 1):
        if i not in indices:
            missing_indices.append(i)
    return missing_indices


def file_name_to_index(files):
    # extract the indices from the file names
    indices = []
    for file in files:
        index = file.split("_")[1]
        # remove the file extension
        index = index.split(".")[0]
        # remove the leading zeros
        index = index.lstrip("0")
        # if the index is empty, it is 0
        if index == "":
            index = "0"
        # ignore copies
        if "copy" in index.lower():
            continue
        indices.append(int(index))

    return sorted(indices)


def get_missing_indices_ranges(missing_files):
    # only extract lowest and highest index of following missing indices
    missing_indices = file_name_to_index(missing_files)
    start_index = missing_indices[0]
    end_index = None
    missing_ranges = []
    for i in range(len(missing_indices) - 1):
        if missing_indices[i + 1] - missing_indices[i] > 1:
            end_index = missing_indices[i]
            missing_ranges.append((start_index, end_index, end_index - start_index + 1))
            start_index = missing_indices[i + 1]

    # After the loop, close the last range
    end_index = missing_indices[-1]
    missing_ranges.append((start_index, end_index, end_index - start_index + 1))
    return missing_ranges


def get_completness_status(
    files_currently_processed, files_to_be_processed, blocks, print_missing_files=True
):
    missing_files, copied_files = compare_file_lists(
        files_currently_processed, files_to_be_processed
    )

    print("Number of files to be processed: ", len(files_to_be_processed))
    print("Copied files: ", copied_files)
    if print_missing_files:
        print("Missing files: ", missing_files)
    print("Number of missing files: ", len(missing_files))

    if missing_files:
        print_block_status(missing_files, blocks)
        return False  # not all files processed
    else:
        assert len(files_currently_processed) == len(files_to_be_processed) - len(
            copied_files
        ), f"Number of files currently processed: {len(files_currently_processed)}, Number of files to be processed: {len(files_to_be_processed)}"
        print("All files processed.")
        return True  # all files processed


def get_blocks(files_currently_processed, files_to_be_processed, num_blocks):
    """
    Divide the missing files into blocks.
    block: (start, end, num_elements, indices)
    """
    missing_files, _ = compare_file_lists(
        files_currently_processed, files_to_be_processed
    )
    if not missing_files:
        print("All files processed.")
        return None

    num_missing_files = len(missing_files)
    ideal_block_size = np.ceil(num_missing_files / num_blocks)
    missing_files_indices = file_name_to_index(missing_files)

    # devide missing files into blocks
    blocks = []
    start_index = missing_files_indices[0]
    end_index = None
    indices = []
    for i in range(num_missing_files - 1):
        indices.append(missing_files_indices[i])
        if len(indices) == ideal_block_size:
            end_index = missing_files_indices[i]
            blocks.append((start_index, end_index, len(indices), indices))
            start_index = missing_files_indices[i + 1]
            indices = []
    # add the last block
    end_index = missing_files_indices[-1]
    indices.append(missing_files_indices[-1])
    blocks.append((start_index, end_index, len(indices), indices))
    return blocks


def print_block_status(missing_files, blocks):
    """
    Returns the start, the current file, the end, and percentage of completness of the current block.
    """
    block_status = []
    missing_files_indices = file_name_to_index(missing_files)
    current_index = missing_files_indices[0]
    for block in blocks:
        if block[0] <= current_index <= block[1]:
            elements_processed_in_block = block[3].index(current_index)
            block_status.append(
                (
                    block[0],
                    current_index,
                    block[1],
                    block[2],
                    elements_processed_in_block,
                )
            )
            # delete elements in missing_files_indices that are in the last block
            missing_files_indices = [i for i in missing_files_indices if i > block[1]]
        if missing_files_indices:
            current_index = missing_files_indices[0]
        else:
            break

    for i, status in enumerate(block_status):
        print(
            f"Block {i+1}: {status[0]} - \033[1;32m {status[1]} \033[0m - {status[2]}, {status[4]}/{status[3]} elements processed. ({status[4]/status[3]*100:.2f}%)"
        )
