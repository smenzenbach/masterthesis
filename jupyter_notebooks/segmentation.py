import os
import sys
import warnings
import yaml
import torch
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import pipeline
from tqdm import tqdm
import numpy as np


# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import nms_grounded, remove_big_box_grounded, remove_small_box_grounded
from utils import load_images_and_combine, extract_index


parser = argparse.ArgumentParser(description="Perform confounder detection")
parser.add_argument(
    "-s",
    "--start",
    default=None,
    type=int,
    help="Image index to start from",
)

parser.add_argument(
    "--id",
    default=None,
    type=str,
    help="ID number for the output directory. If not given the next available ID will be used.",
)


parser.add_argument(
    "-d",
    "--device",
    default="cuda",
    type=str,
    help="Device to use for the model",
)


def run(
    img_path,
    custom_vocab,
    detection_model,
    processor,
    segmentation_pipe,
    device,
    savepath=None,
):
    image = Image.open(img_path)
    text = ""
    for vocab in custom_vocab:
        text += f"a {vocab.lower()} . "

    inputs = processor(image, text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detection_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    """
    Results is a list of dict with the following structure:
    [
        {
            'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
            'labels': ['car', 'tire', 'tire', 'tire'], 
            'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                            [1392.4701,  554.4064, 1628.6133,  777.5872],
                            [ 436.1182,  621.8940,  676.5255,  851.6897],
                            [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
        }
    ]
    Boxes are in (x1, y1, x2, y2) format.
    """

    # Post-process the results
    results = nms_grounded(results)
    results = remove_big_box_grounded(results, image.size, 0.5)
    results = remove_small_box_grounded(results, image.size, 0.06)

    # Create a blank mask for the original image
    original_mask = Image.new("L", image.size, 0)

    for box in results[0]["boxes"].cpu().numpy():
        # pre process the image for segmentation
        # crop the image to the bounding box of the object
        cropped_image = image.crop(box)
        # run the segmentation model
        pillow_mask = segmentation_pipe(cropped_image, return_mask=True)

        # Resize the segmentation mask to the size of the bounding box
        x1, y1, x2, y2 = map(int, box)
        resized_mask = pillow_mask.resize((x2 - x1, y2 - y1))

        # Create a mask of the same size as the original image, filled with 0 (background)
        temp_mask = Image.new("L", image.size, 0)

        # Paste the resized mask into the correct position on the temp_mask
        temp_mask.paste(resized_mask, (x1, y1))

        # Combine the temp_mask with the original_mask using the maximum value (logical OR)
        original_mask = Image.composite(temp_mask, original_mask, temp_mask)

    # Save the mask or display it
    if savepath:
        original_mask.save(savepath)


if __name__ == "__main__":

    args = parser.parse_args()

    # set device
    if args.device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        warnings.warn("CUDA is not available. Using CPU instead.")
    else:
        device = args.device

    # Load the YAML file into a dictionary
    dataset_yaml_file = "/workspace/home/data/datasets.yaml"
    with open(dataset_yaml_file, "r") as file:
        datasets = yaml.safe_load(file)

    # Load images
    # get dataset
    dataset_key = "ISIC2019"
    dataset = datasets[dataset_key]

    # load the ground truth, predicted segmentations and images
    images_path = f"/workspace/home/data/{datasets[dataset_key]['dataset_dir']}/raw"

    images = load_images_and_combine(
        dir_path=images_path,
        confounder_dir=dataset["confounder_dir"],
        no_confounder_dir=dataset["no_confounder_dir"],
    )

    # build grounding dino from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        device
    )

    pipe = pipeline(
        "image-segmentation",
        model="briaai/RMBG-1.4",
        trust_remote_code=True,
        device=device,
    )

    for i, image_dict in tqdm(enumerate(images)):

        image_path = image_dict["meta"]["file_path"]
        image_name = image_dict["meta"]["file_name"]
        # Check for cats and remote controls
        # VERY important: text queries need to be lowercased + end with a dot
        custom_vocabulary = dataset["custom_vocabulary"]

        # skip images before the start index
        if args.start is not None and extract_index(image_name) < args.start:
            continue

        savedir = f"/workspace/home/data/{datasets[dataset_key]['dataset_dir']}/seg_groundedRMBG"
        os.makedirs(savedir, exist_ok=True)
        savepath = f"{savedir}/{image_dict['meta']['file_name']}"

        # skip already processed images
        if os.path.exists(savepath):
            print(f"\033[31m Already segmented {savepath} \033[0m")
            continue

        run(
            image_path,
            custom_vocabulary,
            grounding_model,
            processor,
            pipe,
            device,
            savepath,
        )
