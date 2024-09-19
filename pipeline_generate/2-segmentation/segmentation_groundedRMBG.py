from calendar import c
from email.mime import image
import os
import re
import sys
import warnings
import yaml
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np


# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import nms_grounded, remove_big_box_grounded, remove_small_box_grounded


def segment_instance_groundedRMBG(
    img_dict,
    custom_vocab,
    detection_model,
    processor,
    segmentation_pipe,
    device=0,
    save_path=None,
):
    image = Image.open(img_dict["image_path"])
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
        pil_mask = segmentation_pipe(cropped_image, return_mask=True)

        # make pil_mask a binary mask
        pil_mask = pil_mask.convert("L").point(lambda x: 255 if x > 128 else 0)

        # Resize the segmentation mask to the size of the bounding box
        x1, y1, x2, y2 = map(int, box)
        resized_mask = pil_mask.resize((x2 - x1, y2 - y1))

        # Create a mask of the same size as the original image, filled with 0 (background)
        temp_mask = Image.new("L", image.size, 0)

        # Paste the resized mask into the correct position on the temp_mask
        temp_mask.paste(resized_mask, (x1, y1))

        # Combine the temp_mask with the original_mask using the maximum value (logical OR)
        original_mask = Image.composite(temp_mask, original_mask, temp_mask)

    # Save the mask or display it
    if save_path:
        original_mask.save(save_path)
    else:
        return original_mask
