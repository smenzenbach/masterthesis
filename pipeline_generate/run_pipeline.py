import subprocess
import argparse
import os

parser = argparse.ArgumentParser(description="Pipeline")

parser.add_argument(
    "--config",
    required=True,
    type=str,
    help="Path to the configuration file",
)

parser.add_argument(
    "-nb",
    "--num_blocks",
    default=None,
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

os.chdir("/workspace/home01/pipeline_generate/1-confounder_classification")

# Run confounder classification
subprocess.run(
    f"python run_confounder_classification.py --config {args.config}",
    shell=True,
    check=True,
)

# Wait for confounder classification to finish

# Run segmentation
# if args.num_blocks is None:
os.chdir("/workspace/home01/pipeline_generate/2-segmentation")
subprocess.run(
    f"python run_segmentation.py --config {args.config} --device {args.devices[0]}",
    shell=True,
    check=True,
)
# else:
#    os.chdir("/workspace/home01/pipeline_generate")
#    subprocess.run(
#        f"python multiprocessing_script.py --config {args.config} --script segmentation --num_blocks {args.num_blocks} --devices {' '.join(map(str, args.devices))}",
#        shell=True,
#        check=True,
#    )

# Wait for segmentation to finish

# Run inpainting
if args.num_blocks is None:
    os.chdir("/workspace/home01/pipeline_generate/3-inpainting")
    subprocess.run(
        f"python /workspace/home01/pipeline_generate/3-inpainting/run_inpainting.py --config {args.config} --device {args.devices[0]}",
        shell=True,
        check=True,
    )
else:
    os.chdir("/workspace/home01/pipeline_generate")
    subprocess.run(
        f"python multiprocessing_script.py --config {args.config} --script inpainting --num_blocks {args.num_blocks} --devices {' '.join(map(str, args.devices))}",
        shell=True,
        check=True,
    )

print("Pipeline finished")
