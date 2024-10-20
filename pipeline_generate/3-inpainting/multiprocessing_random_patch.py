import os, sys
import multiprocessing
import subprocess
import threading
import argparse
import yaml
import time
import toml

# Add the project root directory to sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)  # Adjust this path if needed
sys.path.append(project_root)

from utils import (
    create_file_names_list,
    get_blocks,
    get_completness_status,
    compare_file_lists,
)

parser = argparse.ArgumentParser(description="Inpainting with iopaint")


parser.add_argument(
    "-nb",
    "--num_blocks",
    default=10,
    type=int,
    help="Number of blocks to split the data into",
)

parser.add_argument(
    "-d",
    "--devices",
    nargs="+",
    default=[0, 1, 2, 3],
    type=int,
    help="Number of devices to run the script on",
)

args = parser.parse_args()

start_time = time.time()


def run_script(start=None, device=None):
    script_string = [
        "python",
        "/workspace/home01/pipeline_generate/3-inpainting/inpaint_non_patch_imgs.py",
    ]
    if start:
        script_string.extend(["-s", str(start)])  # Convert start to string
    if device is not None:
        script_string.extend(["-d", str(device)])  # Convert device to string
    subprocess.run(script_string, check=True)


def print_status(
    processes, files_currently_processed_path, files_to_be_processed, blocks
):
    while any(p.is_alive() for p in processes):
        files_currently_processed = create_file_names_list(
            files_currently_processed_path
        )
        complete = get_completness_status(
            files_currently_processed,
            files_to_be_processed,
            blocks,
            print_missing_files=False,
        )
        if complete:
            # stop all processes
            for p in processes:
                p.terminate()
        # print current duration of the script in 0:00:00 format
        print(
            f"Time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}"
        )
        for i, p in enumerate(processes):
            print(f"process {i} is alive: {p.is_alive()}")
        time.sleep(60)  # Wait for 5 minutes (300 seconds)


dataset_yaml_file = "/workspace/home/data/datasets.yaml"
with open(dataset_yaml_file, "r") as file:
    datasets = yaml.safe_load(file)
dataset = datasets["ISIC2019"]

# load images
segmentation_masks_path = dataset["gt_seg_path"]
segmentation_masks = create_file_names_list(segmentation_masks_path)
all_images = create_file_names_list(dataset["raw_path"])
# get images that are not yet segmented
images_list = compare_file_lists(segmentation_masks, all_images)[0]

inpainted_folder_path = (
    f"/workspace/home/pipeline_outputs/ISIC2019/random_inpaintings/inpainting"
)

target_files = create_file_names_list(inpainted_folder_path)
source_files = images_list

target_path = inpainted_folder_path


blocks = get_blocks(target_files, source_files, args.num_blocks)

num_processes_per_device = len(blocks) // len(args.devices)

# Create a process for block
processes = []
device = args.devices[0]
for i, block in enumerate(blocks):
    if i % num_processes_per_device == 0:
        device = args.devices[(i // num_processes_per_device) - 1]
    print(
        f"Running block {block[0]} to {block[1]} ({block[2]} images) on device {device}"
    )
    p = multiprocessing.Process(target=run_script, args=(block[0], device))
    processes.append(p)
    p.start()

# Start a thread to print the status every 5 minutes
status_thread = threading.Thread(
    target=print_status,
    args=(processes, target_path, source_files, blocks),
)
status_thread.start()

# Wait for all processes to complete
for p in processes:
    p.join()

# Ensure the status thread also completes once all processes are done
status_thread.join()

print("All scripts have finished.")


# python multiprocessing_script.py --config [config_file_path] --scipt [pipline step] --num_blocks 9 --devices 0 1 2 3
# python multiprocessing_script.py --config /workspace/home01/pipeline_configs/config_0021.toml --script inpainting  --num_blocks 9 --devices 0 1 2 3
