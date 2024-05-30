import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw


import cv2
import numpy as np

from home_robot.core.abstract_perception import PerceptionModule  # noqa: E402
from home_robot.core.interfaces import Observations  # noqa: E402
from home_robot.perception.detection.utils import (  # noqa: E402
    filter_depth,
    overlay_masks,
)

import functools
from ..fvlm.demo_utils import input_utils
from ..fvlm.demo_utils import vis_utils
from ..fvlm.demo_utils import mask_utils
import jax
import tensorflow as tf
import tqdm
from .utils import clip_utils

#Test for only loading graph once
#tf.compat.v1.disable_eager_execution()
import time

_MAX_BOXES_TO_DRAW = 25
_MAX_NUM_CLS = 1204
_MIN_SCORE_THRESH = 0.01

PARENT_DIR = Path(__file__).resolve().parent

_TOKENIZER_CHECKPOINT_PATH = str(
    PARENT_DIR / "dito" / "checkpoints" / "tokenizer" / "dito_tokenizer.model"
)
_TEXT_MODEL_CHECKPOINT_PATH = str(
    PARENT_DIR / "dito" / "checkpoints" / "text_model"
)
_EMBEDDING_PATH = str(
    PARENT_DIR / "dito" / "embeddings" / "bg_empty_embed.npy"
)
_MODEL_CHECKPOINT_PATH = str(
    PARENT_DIR / "dito" / "checkpoints" / "vit-large"
)


class DitoPerception(PerceptionModule):
    def __init__(
        self,
        config_file=None,
        custom_vocabulary: List[str] = "['colorful patch', 'colorful sticker', 'colorful circle', 'colorful cuttet circle']",
        sem_gpu_id=None,
        checkpoint_file: str = None,
        verbose=False,
    ):
        """Load trained LSeg model for inference.

        Arguments:
            config_file: path to model config
            custom_vocabulary: if vocabulary="custom", this should be a comma-separated
             list of classes (as a single string)
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            checkpoint_file: path to model checkpoint
            verbose: whether to print out debug information
        """

        # Building Dito model for inference
        #Get tokenizer and text model
        self.tokenizer = clip_utils.get_tokenizer(_TOKENIZER_CHECKPOINT_PATH)
        with tf.device('/cpu:0'):
            self.text_model = tf.saved_model.load(_TEXT_MODEL_CHECKPOINT_PATH)

        self.background_embedding, self.empty_embeddings = np.load(_EMBEDDING_PATH)
        self.background_embedding = self.background_embedding[np.newaxis, Ellipsis]
        self.empty_embeddings = self.empty_embeddings[np.newaxis, Ellipsis]
        self.tile_empty_embeddings = np.tile(
            self.empty_embeddings, (_MAX_NUM_CLS - len(custom_vocabulary) - 1, 1)
        )

        self.custom_vocabulary = custom_vocabulary

        #Create Text encodings
        class_clip_features = []
        with tf.device('/cpu:0'):
            for cls_name in tqdm.tqdm(self.custom_vocabulary, total=len(self.custom_vocabulary)):
                cls_feat = clip_utils.tokenize_pad_fn(self.tokenizer, self.text_model, cls_name)
                class_clip_features.append(cls_feat)
        text_embeddings = np.concatenate(class_clip_features, axis=0)
        
        # Concatenate 'background' and 'empty' embeddings.
        text_embeddings = np.concatenate(
            (self.background_embedding, text_embeddings, self.tile_empty_embeddings), axis=0
        )
        self.text_embeddings = text_embeddings[np.newaxis, Ellipsis]

        self.parser_fn = input_utils.get_rovit_parser()

        self.model = tf.saved_model.load(_MODEL_CHECKPOINT_PATH)


    def reset_vocab(self, new_vocab: List[str]):
        """Resets the vocabulary of the model allowing you to change detection on
        the fly. Note that previous vocabulary is not preserved.
        Args:
            new_vocab: list of strings representing the new vocabulary
        """
        self.custom_vocabulary = new_vocab

        self.tile_empty_embeddings = np.tile(
            self.empty_embeddings, (_MAX_NUM_CLS - len(self.custom_vocabulary) - 1, 1)
        )

        #Create Text encodings
        class_clip_features = []
        with tf.device('/cpu:0'):
            for cls_name in tqdm.tqdm(self.custom_vocabulary, total=len(self.custom_vocabulary)):
                cls_feat = clip_utils.tokenize_pad_fn(self.tokenizer, self.text_model, cls_name)
                class_clip_features.append(cls_feat)
        text_embeddings = np.concatenate(class_clip_features, axis=0)
        
        # Concatenate 'background' and 'empty' embeddings.
        text_embeddings = np.concatenate(
            (self.background_embedding, text_embeddings, self.tile_empty_embeddings), axis=0
        )
        self.text_embeddings = text_embeddings[np.newaxis, Ellipsis]

    def predict(
        self,
        obs: Observations,
        depth_threshold: Optional = None,
        draw_instance_predictions: bool = False,
    ):
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in RGB order)
            obs.depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """

        if draw_instance_predictions:
            raise NotImplementedError

        #Create image input data
        data = self.parser_fn({'image': obs.rgb, 'source_id': np.array([0])})
        np_data = jax.tree_map(lambda x: x.numpy()[np.newaxis, Ellipsis], data)
        np_data['text'] = self.text_embeddings
        np_data['image'] = np_data.pop('images')
        labels = np_data.pop('labels')
        image = np_data['image']

        #Run model
        output = self.model(np_data)
        
        detection_scores = np.squeeze(output['detection_scores'], axis=(0,))

        count_over_threshold = np.sum(detection_scores > _MIN_SCORE_THRESH) -1

        if count_over_threshold < 0:
            obs.semantic = np.zeros((obs.rgb.shape[0], obs.rgb.shape[1])).astype(int)
            if obs.task_observations is None:
                obs.task_observations = dict()
            obs.task_observations["semantic_frame"] = None
            return obs

        # Scale the boxes to normalized image space.
        detection_boxes = np.squeeze(output['detection_boxes'], axis=(0,))[
            0:count_over_threshold
        ]
        detection_classes = np.squeeze(
            output['detection_classes'].astype(np.int32), axis=(0,)
        )[0:count_over_threshold]

        detection_boxes = detection_boxes * 0.625
        segmentations = None
        if 'detection_masks' in output:
            instance_masks = np.squeeze(output['detection_masks'], axis=(0,))[
                0:count_over_threshold
            ]
            # Paste the masks to image space
            # Change the format to xywh.
            y1, x1, y2, x2 = np.split(detection_boxes, 4, -1)
            xywh_detection_boxes = np.concatenate([x1, y1, x2 - x1, y2 - y1], axis=-1)
            segmentations = mask_utils.paste_instance_masks(
                instance_masks,
                xywh_detection_boxes,
                image_height=obs.rgb.shape[0],
                image_width=obs.rgb.shape[1],
            )

        if depth_threshold is not None and obs.depth is not None:
            segmentations = np.array(
                [
                    filter_depth(mask, obs.depth, depth_threshold)
                    for mask in segmentations
                ]
            )

        #Put it all in one semantic map
        semantic_map, instance_map = overlay_masks(
            segmentations, detection_classes, (obs.rgb.shape[0], obs.rgb.shape[1])
        )
        semantic_map[semantic_map != 0] -= 1
        obs.semantic = semantic_map.astype(int)
        obs.instance = instance_map.astype(int)
        if obs.task_observations is None:
            obs.task_observations = dict()
        obs.task_observations["instance_map"] = instance_map
        obs.task_observations["instance_classes"] = detection_classes
        obs.task_observations["instance_scores"] = detection_scores
        obs.task_observations["semantic_frame"] = None
        return obs
