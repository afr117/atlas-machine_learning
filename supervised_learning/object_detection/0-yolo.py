#!/usr/bin/env python3
"""Yolo v3 Object Detection Initialization"""

import tensorflow as tf
import numpy as np


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor

        Args:
            model_path (str): path to where a Darknet Keras model is stored
            classes_path (str): path to where class names are stored
            class_t (float): box score threshold for the initial filtering step
            nms_t (float): IOU threshold for non-max suppression
            anchors (np.ndarray): anchor boxes with shape (outputs, anchor_boxes, 2)
        """
        # Load the Darknet Keras model
        self.model = tf.keras.models.load_model(model_path)

        # Load class names
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Assign thresholds and anchors
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
