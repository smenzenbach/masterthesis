import os
import sys
import torch
from PIL import Image
from torchvision import transforms


# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoModelForImageSegmentation,
)
from transformers import pipeline

from utils import nms_grounding, remove_big_box_grounding, remove_small_box_grounding


def segment_instance_groundingRMBG2(
    img_dict,
    custom_vocab,
    detection_model,
    processor,
    segmentation_model,
    device=0,
    save_path=None,
):
    # Load and process the input image
    image = Image.open(img_dict["file_path"])
    text = " ".join([f"a {vocab.lower()} ." for vocab in custom_vocab])
    inputs = processor(image, text, return_tensors="pt").to(device)

    # Run object detection
    with torch.no_grad():
        outputs = detection_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    # Post-process the detection results
    results = nms_grounding(results)
    results = remove_big_box_grounding(results, image.size, 0.5)
    results = remove_small_box_grounding(results, image.size, 0.06)

    # Create a blank mask for the original image
    original_mask = Image.new("L", image.size, 0)
    bounding_box_on_image = image.convert("RGBA")

    # Set up image transformation
    image_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    for box in results[0]["boxes"].cpu().numpy():
        # Crop the image to the bounding box
        cropped_image = image.crop(box)

        # Preprocess the image
        input_images = transform_image(cropped_image).unsqueeze(0).to(device)

        # Run segmentation with the model
        with torch.no_grad():
            preds = segmentation_model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pil_mask = transforms.ToPILImage()(pred)

        # Make the segmentation mask binary
        pil_mask = pil_mask.convert("L").point(lambda x: 255 if x > 128 else 0)

        # Resize the mask to the bounding box size
        x1, y1, x2, y2 = map(int, box)
        resized_mask = pil_mask.resize((x2 - x1, y2 - y1))

        # Create a mask of the same size as the original image, filled with 0 (background)
        temp_mask = Image.new("L", image.size, 0)
        temp_mask.paste(resized_mask, (x1, y1))

        # Combine the temp_mask with the original_mask
        original_mask = Image.composite(temp_mask, original_mask, temp_mask)

        # Draw the bounding box on the image
        bounding_box_on_image = Image.alpha_composite(
            bounding_box_on_image,
            Image.new("RGBA", bounding_box_on_image.size, (255, 255, 255, 0)),
        )

    assert (
        original_mask.size[0] == image.size[0]
        and original_mask.size[1] == image.size[1]
    )

    # Save or return the results
    if save_path:
        original_mask.save(save_path)
    else:
        return original_mask, bounding_box_on_image


def setup_processor_model_pipeline2(device=0):
    # Load the detection model and processor
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        f"cuda:{device}"
    )

    # Load the segmentation model
    segmentation_model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0", trust_remote_code=True
    ).to(f"cuda:{device}")
    segmentation_model.eval()

    return processor, grounding_model, segmentation_model
