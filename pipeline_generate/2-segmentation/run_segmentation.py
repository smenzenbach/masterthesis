import os
import sys
import argparse
import yaml
import toml

# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import (
    get_images_from_csv,
    get_images_from_list,
    create_file_names_list_from_csv,
    create_file_names_list,
    extract_index,
    compare_file_lists,
)
from segmentation_dito import segment_instance_dito
from segmentation_groundedRMBG import segment_instance_groundedRMBG

parser = argparse.ArgumentParser(description="Perform segmentation")

parser.add_argument(
    "--config",
    required=True,
    type=str,
    help="Path to the configuration file",
)

parser.add_argument(
    "-s",
    "--start",
    default=None,
    type=int,
    help="Image index to start from",
)

parser.add_argument(
    "-d",
    "--device",
    default=0,
    type=int,
    help="GPU device number",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Load the configuration file
    config = toml.load(args.config)

    # check if existing segmentation should be used
    if "use_existing" in config["segmentation"]:
        print("Using existing segmentation. Exiting the program.")
        sys.exit()


    # get id that is between "_" and "." in the config file name
    id_tag = int(args.config.split("_")[-1].split(".")[0])

    # Load YAML file into a dictionary
    dataset_name = config["dataset"]["name"]
    dataset_yaml_file = config["dataset"]["yaml_file"]
    with open(dataset_yaml_file, "r") as file:
        datasets = yaml.safe_load(file)

    # get dataset
    dataset = datasets[dataset_name]

    # get images
    raw_images_path = dataset["raw_path"]

    # get csv path with confounder classification results
    # check which confounder classification results should be used
    if "use_existing" in config["confounder_classification"]:
        confounder_pred_id = config["confounder_classification"]["use_existing"]
        if confounder_pred_id == "gt":
            csv_path = dataset["gt_label_path"]
        else:
            csv_path = f"/workspace/home/pipeline_outputs/{config["dataset"]["name"]}/{confounder_pred_id}/confounder_pred.csv"
    else:
        csv_path = f"/workspace/home/pipeline_outputs/{config["dataset"]["name"]}/{id_tag}/confounder_pred.csv"

    # get images that should be segmented
    if args.start is None:
        images = get_images_from_csv(
            csv_path, raw_images_path, filter_func=lambda row: row[1] == "1"
        )
    else:
        # get list of image names from the directory
        image_names = create_file_names_list_from_csv(
            csv_path, filter_func=lambda row: row[1] == "1"
        )
        # get the image names from the start index (the start index in the name)
        image_names = [
            name for name in image_names if extract_index(name) >= args.start
        ]
        # get the images from the list of image names
        images = get_images_from_list(image_names, raw_images_path)

    # set output dir for segmentation results
    output_dir = f"/workspace/home/pipeline_outputs/{config["dataset"]["name"]}/{id_tag}/seg"

    # check if the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(
            "Output directory already exists. Check if there are still unsegmented images."
        )
        # check if all files are already segmented
        already_seg_images = create_file_names_list(output_dir)
        to_be_segmented_list = create_file_names_list_from_csv(
            csv_path, filter_func=lambda row: row[1] == "1"
        )
        missing_files, _ = compare_file_lists(already_seg_images, to_be_segmented_list)
        if len(missing_files) == 0:
            print("All images are already segmented.")
            sys.exit(0)
        else:
            print(f"Found {len(missing_files)} images to process.")
            # only process the missing files
            images = get_images_from_list(missing_files, raw_images_path)

    # set custom vocabulary for ov object detection model
    custom_vocabulary =  config["segmentation"]["custom_voabulary"]

    # retrieve used model
    model = config["segmentation"]["model"]

    # setup segmentation model
    if model == "groundedRMBG":
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from transformers import pipeline

        # build grounding dino from huggingface
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(f"cuda:{args.device}")

        pipe = pipeline(
            "image-segmentation",
            model="briaai/RMBG-1.4",
            trust_remote_code=True,
            device=f"cuda:{args.device}",
        )
    elif model == "dito":
        from segmentation_dito import setup_model
        import tensorflow as tf

        model, text_embeddings, parser_fn = setup_model(custom_vocabulary, args.device)
        confidence_threshold = config["segmentation.dito"]["confidence_threshold"]
    else:
        raise ValueError(f"Unknown segmentation model: {model}")

    # run segmentation
    for i, img_dict in enumerate(images):
        # set path were the segmented image should be saved
        save_path = f"{output_dir}/{img_dict['file_name']}"
        # skip already processed images
        if os.path.exists(save_path):
            print(f"\033[31m Already segmented {save_path} \033[0m")
            continue
        elif model == "groundedRMBG":
            # TODO: implement customizable threshold
            segment_instance_groundedRMBG(
                img_dict,
                custom_vocabulary,
                grounding_model,
                processor,
                pipe,
                args.device,
                save_path,
            )
        elif model == "dito":
            with tf.device(f"/GPU:{args.device}"):
                output, vis_image = segment_instance_dito(
                    img_dict,
                    custom_vocabulary,
                    confidence_threshold,
                    model,
                    parser_fn,
                    text_embeddings,
                    # device=args.device,
                    save_path=save_path,
                )
