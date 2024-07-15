import os
import gc
import random
import requests
from typing import List, Optional, Dict, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
import cv2
import supervision as sv
import tensorflow as tf
import torch
import torchvision
from transformers import AutoModelForMaskGeneration, AutoProcessor
from transformers import pipeline
from inference.models import YOLOWorld

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# ------------------------- #
#       result utils        #
# ------------------------- #


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


# ------------------------- #
#        plot utils         #
# ------------------------- #


def annotate(
    image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]
) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(
            image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2
        )
        cv2.putText(
            image_cv2,
            f"{label}: {score:.2f}",
            (box.xmin, box.ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color.tolist(),
            2,
        )

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def yolo_plot(
    image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]
):

    MASK_ANNOTATOR = sv.MaskAnnotator()
    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=3)
    LABEL_ANNOTATOR = sv.LabelAnnotator(
        text_thickness=2, text_scale=1, text_color=sv.Color.BLACK
    )
    annotated_image = image.copy()
    annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detection_results)

    if detection_results.mask is not None:
        annotated_image = MASK_ANNOTATOR.annotate(annotated_image, detection_results)
        for mask in detection_results.mask:
            annotated_image = overlay(
                annotated_image, mask, color=(105, 183, 228), alpha=0.5, resize=None
            )
    else:
        annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(
            annotated_image, detection_results
        )

    sv.plot_image(annotated_image, (6, 6))


def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None,
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis("off")
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")
    plt.show()


def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        "aliceblue",
        "antiquewhite",
        "aqua",
        "aquamarine",
        "azure",
        "beige",
        "bisque",
        "black",
        "blanchedalmond",
        "blue",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgreen",
        "darkgrey",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkslategrey",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dimgrey",
        "dodgerblue",
        "firebrick",
        "floralwhite",
        "forestgreen",
        "fuchsia",
        "gainsboro",
        "ghostwhite",
        "gold",
        "goldenrod",
        "gray",
        "green",
        "greenyellow",
        "grey",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        "lightblue",
        "lightcoral",
        "lightcyan",
        "lightgoldenrodyellow",
        "lightgray",
        "lightgreen",
        "lightgrey",
        "lightpink",
        "lightsalmon",
        "lightseagreen",
        "lightskyblue",
        "lightslategray",
        "lightslategrey",
        "lightsteelblue",
        "lightyellow",
        "lime",
        "limegreen",
        "linen",
        "magenta",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        "navajowhite",
        "navy",
        "oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "rebeccapurple",
        "red",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        "seashell",
        "sienna",
        "silver",
        "skyblue",
        "slateblue",
        "slategray",
        "slategrey",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        "white",
        "whitesmoke",
        "yellow",
        "yellowgreen",
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))


def plot_detections_plotly(
    image: np.ndarray,
    detections: List[DetectionResult],
    class_colors: Optional[Dict[str, str]] = None,
) -> None:
    # If class_colors is not provided, generate random colors for each class
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {}
        for i in range(num_detections):
            class_colors[i] = colors[i]

    fig = px.imshow(image)

    # Add bounding boxes
    shapes = []
    annotations = []
    for idx, detection in enumerate(detections):
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask

        polygon = mask_to_polygon(mask)

        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in polygon] + [polygon[0][0]],
                y=[point[1] for point in polygon] + [polygon[0][1]],
                mode="lines",
                line=dict(color=class_colors[idx], width=2),
                fill="toself",
                name=f"{label}: {score:.2f}",
            )
        )

        xmin, ymin, xmax, ymax = box.xyxy
        shape = [
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=xmin,
                y0=ymin,
                x1=xmax,
                y1=ymax,
                line=dict(color=class_colors[idx]),
            )
        ]
        annotation = [
            dict(
                x=(xmin + xmax) // 2,
                y=(ymin + ymax) // 2,
                xref="x",
                yref="y",
                text=f"{label}: {score:.2f}",
            )
        ]

        shapes.append(shape)
        annotations.append(annotation)

    # Update layout
    button_shapes = [dict(label="None", method="relayout", args=["shapes", []])]
    button_shapes = button_shapes + [
        dict(label=f"Detection {idx+1}", method="relayout", args=["shapes", shape])
        for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [
        dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])
    ]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        updatemenus=[dict(type="buttons", direction="up", buttons=button_shapes)],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Show plot
    fig.show()


# ------------------------- #
#           utils           #
# ------------------------- #


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(
    polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image


def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def refine_masks(
    masks: torch.BoolTensor, polygon_refinement: bool = False
) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


# ------------------------- #
#        yolo + sam         #
# ------------------------- #


def detect_yolo_world(
    image: Image.Image,
    yolo_model,
    confidence: float = 0.0025,
    labels: List[str] = ["colorful cutted circle", "colorful circle"],
):

    classes = labels
    yolo_model.set_classes(classes)

    results = yolo_model.infer(image, confidence=confidence)
    detections = sv.Detections.from_inference(results)
    return detections


def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None,
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = detection_results.xyxy  # get_boxes(detection_results)
    # Convert the xyxy array to a list of lists of lists of floats
    boxes = [[[float(coord) for coord in box] for box in boxes]]
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes,
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    detection_results.mask = np.array(masks)

    # for detection_result, mask in zip(detection_results, masks):
    #    detection_result.mask = mask

    return detection_results


def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    iou_threshold: float = 0.8,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Perform detection and segmentation given a custom vocabulary on an image using YOLO and SAM.
    """

    if isinstance(image, str):
        image = cv2.imread(image)

    detections = detect_yolo_world(image, confidence=threshold, labels=labels)
    detections = nms_yolo(detections, iou_threshold)
    if len(detections.xyxy) == 0:
        print("no patch found")
    else:
        detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections


# ------------------------- #
#         show masks        #
# ------------------------- #


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()


def masks_on_image(
    raw_image, masks, show=True, save=False, folderpath=None, filename=None
):
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    seed = 42
    for mask in masks:
        np.random.seed(seed)
        show_mask(mask, ax=ax, random_color=True)
        seed += 7
    plt.axis("off")
    if show:
        plt.show()

    if save:
        if folderpath is None:
            raise TypeError("Please provide folderpath")
        if filename is None:
            raise TypeError("Please provide filename")

        # Ensure the folder exists
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        # Save the figure
        save_path = os.path.join(folderpath, filename)
        plt.savefig(save_path, format="jpg", bbox_inches="tight", pad_inches=0)
        print(f"save img {filename}")
        if not show:
            plt.close()
    del mask
    gc.collect()


# ------------------------------- #
#  remove overlapping detections  #
# ------------------------------- #


def nms_yolo(detection_result: List[Dict[str, Any]], iou_threshold: float = 0.5):
    """
    Perform non-maximum suppression on the detection results.
    """

    # check if detection_result is empty
    if len(detection_result.xyxy) == 0:
        return detection_result

    boxes = detection_result.xyxy
    scores = detection_result.confidence

    # Convert the bounding boxes to the format required by torchvision.ops.nms
    boxes = torch.tensor(boxes)
    scores = torch.tensor([scores]).squeeze()

    # Perform non-maximum suppression
    nms_indices = torchvision.ops.nms(boxes, scores, iou_threshold).tolist()

    # Filter out the detections that were not selected by non-maximum suppression
    detection_result.xyxy = detection_result.xyxy[nms_indices]
    detection_result.confidence = detection_result.confidence[nms_indices]
    detection_result.class_id = detection_result.class_id[nms_indices]
    detection_result.data["class_name"] = detection_result.data["class_name"][
        nms_indices
    ]

    return detection_result


def nms_dito(
    detection_result: Dict[str, Any],
    iou_threshold: float = 0.5,
    max_output_size: int = 100,
) -> Dict[str, Any]:
    """
    Perform non-maximum suppression on the detection results.
    """

    # Check if detection_result is empty
    if detection_result["detection_boxes"].shape[0] == 0:
        return detection_result

    boxes = tf.squeeze(detection_result["detection_boxes"])
    scores = tf.squeeze(detection_result["detection_scores"])

    # Perform non-maximum suppression
    nms_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size, iou_threshold=iou_threshold
    ).numpy()

    # Filter out the detections that were not selected by non-maximum suppression
    detection_result["detection_boxes"] = tf.gather(
        detection_result["detection_boxes"], nms_indices, axis=1
    )
    detection_result["detection_scores"] = tf.gather(
        detection_result["detection_scores"], nms_indices, axis=1
    )
    detection_result["detection_classes"] = tf.gather(
        detection_result["detection_classes"], nms_indices, axis=1
    )
    detection_result["detection_masks"] = tf.gather(
        detection_result["detection_masks"], nms_indices, axis=1
    )
    detection_result["num_detections"] = tf.constant([len(nms_indices)], dtype=tf.int32)

    return detection_result


# ----------------------------------- #
#  remove picture-filling detections  #
# ----------------------------------- #


def remove_big_box_yolo(
    detection_result: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    area_threshold: float = 0.9,
):
    """
    Remove detections that fill more than x percent of the image. The percentage is given by the area_threshold.
    """

    # check if detection_result is empty
    if len(detection_result.xyxy) == 0:
        return detection_result

    boxes = detection_result.xyxy

    # Filter out the detections that fill the entire image
    filtered_indices = []
    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        bb_area = width * height
        img_area = image_shape[0] * image_shape[1]
        if bb_area / img_area < area_threshold:
            filtered_indices.append(idx)

    # Filter out the detections that were not selected
    detection_result.xyxy = detection_result.xyxy[filtered_indices]
    detection_result.confidence = detection_result.confidence[filtered_indices]
    detection_result.class_id = detection_result.class_id[filtered_indices]
    detection_result.data["class_name"] = detection_result.data["class_name"][
        filtered_indices
    ]

    return detection_result


def remove_big_box_dito(
    detection_result: Dict[str, Any],
    image_shape: Tuple[int, int],
    area_threshold: float = 0.9,
):
    """
    Remove detections that fill more than x percent of the image. The percentage is given by the area_threshold.
    """

    # Check if detection_result is empty
    if detection_result["detection_boxes"].shape[0] == 0:
        return detection_result

    boxes = tf.squeeze(detection_result["detection_boxes"]).numpy()

    # Check if there's only one bounding box
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, axis=0)

    # Filter out the detections that fill the entire image
    filtered_indices = []
    for idx, box in enumerate(boxes):
        ymin, xmin, ymax, xmax = box
        width = xmax - xmin
        height = ymax - ymin
        bb_area = width * height
        img_area = image_shape[0] * image_shape[1]
        if bb_area / img_area < area_threshold:
            filtered_indices.append(idx)

    if len(filtered_indices) == 0:
        return detection_result

    # Filter out the detections that were not selected
    detection_result["detection_boxes"] = tf.gather(
        detection_result["detection_boxes"], filtered_indices, axis=1
    )
    detection_result["detection_scores"] = tf.gather(
        detection_result["detection_scores"], filtered_indices, axis=1
    )
    detection_result["detection_classes"] = tf.gather(
        detection_result["detection_classes"], filtered_indices, axis=1
    )
    detection_result["detection_masks"] = tf.gather(
        detection_result["detection_masks"], filtered_indices, axis=1
    )
    detection_result["num_detections"] = tf.constant(
        [len(filtered_indices)], dtype=tf.int32
    )

    return detection_result
