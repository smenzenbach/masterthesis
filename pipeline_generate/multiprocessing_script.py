import os, sys
import multiprocessing
import subprocess
import threading
import argparse
import yaml
import time

# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import (
    create_file_names_list,
    create_file_names_list_from_csv,
    get_blocks,
    get_completness_status,
)

parser = argparse.ArgumentParser(description="Inpainting with iopaint")

parser.add_argument(
    "--id",
    type=str,
    required=True,
    help="ID number of experiment",
)

parser.add_argument("--script", type=str, required=True, help="script to run")

parser.add_argument(
    "-nb",
    "--num_blocks",
    default=10,
    type=int,
    help="Number of blocks to split the data into",
)

parser.add_argument(
    "--devices",
    nargs="+",
    default=[2],
    type=int,
    help="Number of devices to run the script on",
)

args = parser.parse_args()

start_time = time.time()


def run_script(script, start=None, device=None):
    script_string = ["python", script, "--id", args.id]
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
        get_completness_status(
            files_currently_processed,
            files_to_be_processed,
            blocks,
            print_missing_files=False,
        )
        # print current duration of the script in 0:00:00 format
        print(
            f"Time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}"
        )
        time.sleep(300)  # Wait for 5 minutes (300 seconds)


# Load the YAML file into a dictionary
dataset_yaml_file = "/workspace/home/data/datasets.yaml"
with open(dataset_yaml_file, "r") as file:
    datasets = yaml.safe_load(file)

dataset_key = "ISIC2019"  # 'Pascal_VOC07' #
dataset = datasets[dataset_key]

if "dito.py" in args.script:
    seg_method = "dito"

    segmentation_path = f"/workspace/home/data/{dataset_key}/seg_{seg_method}/{args.id}"
    sgl_pred_csv_path = (
        f"/workspace/home/pipeline_outputs/{dataset_key}/{args.id}/sgl_pred.csv"
    )
    if not os.path.exists(segmentation_path):
        target_files = []
    else:
        target_files = create_file_names_list(segmentation_path)
    source_files = create_file_names_list_from_csv(
        sgl_pred_csv_path, lambda row: row[1] == "1"
    )
    target_path = segmentation_path
elif "iopaint_copy.py" in args.script:
    # TODO: adapt folder paths (with config file?)
    seg_method = "groundedRMBG-1.4"
    inpainted_folder_path = f"{dataset['inpainted_path']}/{args.id}"
    seg_folder_path = f"/workspace/home/data/{dataset_key}/seg_{seg_method}/{args.id}"
    # f"{dataset['seg_path']}_dito/{args.id}"

    if not os.path.exists(inpainted_folder_path):
        os.makedirs(inpainted_folder_path)

    target_files = create_file_names_list(inpainted_folder_path)
    source_files = create_file_names_list(seg_folder_path)

    target_path = inpainted_folder_path


blocks = get_blocks(target_files, source_files, args.num_blocks)

num_processes_per_device = len(blocks) // len(args.devices)

# Create a process for block
processes = []
device = args.devices[0]
for i, block in enumerate(blocks):
    if i % num_processes_per_device == 0:
        device = args.devices[(i // num_processes_per_device) - 1]
    print(f"Running block {block[0]} to {block[1]} on device {device}")
    p = multiprocessing.Process(target=run_script, args=(args.script, block[0], device))
    processes.append(p)
    p.start()

# Start a thread to print the status every 5 minutes
status_thread = threading.Thread(
    target=print_status,
    args=(processes, target_path, source_files, blocks, target_path),
)
status_thread.start()

# Wait for all processes to complete
for p in processes:
    p.join()

# Ensure the status thread also completes once all processes are done
status_thread.join()

print("All scripts have finished.")


# python multiprocessing_script.py --id 0020  --script /workspace/home01/playground/iopaint_copy.py --num_blocks 9 --devices 0 1 2 3
