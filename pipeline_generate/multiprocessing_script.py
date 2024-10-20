import os, sys
import multiprocessing
import subprocess
import threading
import argparse
import yaml
import time
import toml

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
    "--config", required=True, type=str, help="Path to the configuration file"
)

parser.add_argument(
    "--script",
    type=str,
    required=True,
    choices=["confounder_classification", "segmentation", "inpainting"],
    help="script to run",
)

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
    script_string = ["python", script, "--config", args.config]
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

        time.sleep(60)  # Wait for 1 minutes


# Load the configuration file
config = toml.load(args.config)

# get id that is between "_" and "." in the config file name
id_tag = args.config.split("/")[-1].split("_", 1)[1].split(".")[0]

# Load YAML file into a dictionary
dataset_name = config["dataset"]["name"]
dataset_yaml_file = config["dataset"]["yaml_file"]
with open(dataset_yaml_file, "r") as file:
    datasets = yaml.safe_load(file)

# get dataset
dataset = datasets[dataset_name]


if args.script == "confounder_classification":
    raise NotImplementedError(
        "Multithreading for confounder classification not implemented yet"
    )
elif args.script == "segmentation":
    segmentation_path = (
        f"/workspace/home/pipeline_outputs/{config['dataset']['name']}/{id_tag}/seg"
    )
    if "use_existing" in config["confounder_classification"]:
        confounder_pred_id = config["confounder_classification"]["use_existing"]
        if confounder_pred_id == "gt":
            csv_path = dataset["gt_label_path"]
        else:
            csv_path = f"/workspace/home/pipeline_outputs/{config['dataset']['name']}/{confounder_pred_id}/confounder_pred.csv"
    else:
        csv_path = f"/workspace/home/pipeline_outputs/{config['dataset']['name']}/{id_tag}/confounder_pred.csv"

    if not os.path.exists(segmentation_path):
        target_files = []
    else:
        target_files = create_file_names_list(segmentation_path)
    source_files = create_file_names_list_from_csv(csv_path, lambda row: row[1] == "1")
    target_path = segmentation_path

    script_name = (
        "/workspace/home01/pipeline_generate/2-segmentation/run_segmentation.py"
    )
elif args.script == "inpainting":
    inpainted_folder_path = f"/workspace/home/pipeline_outputs/{config['dataset']['name']}/{id_tag}/inpainting"

    if "use_existing" in config["segmentation"]:
        segmentation_id = config["segmentation"]["use_existing"]
        if segmentation_id == "gt":
            segmentation_masks_path = dataset["gt_seg_path"]
        else:
            segmentation_masks_path = f"/workspace/home/pipeline_outputs/{config['dataset']['name']}/{segmentation_id}/seg"
    else:
        segmentation_masks_path = (
            f"/workspace/home/pipeline_outputs/{config['dataset']['name']}/{id_tag}/seg"
        )

    if not os.path.exists(inpainted_folder_path):
        os.makedirs(inpainted_folder_path)

    target_files = create_file_names_list(inpainted_folder_path)
    source_files = create_file_names_list(segmentation_masks_path)

    target_path = inpainted_folder_path

    script_name = "/workspace/home01/pipeline_generate/3-inpainting/run_inpainting.py"


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
    p = multiprocessing.Process(target=run_script, args=(script_name, block[0], device))
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
