#!/usr/bin/env python3
"""YOLO v3 Object Detection"""
import tensorflow as tf
import numpy as np


class Yolo:
    """Yolo class to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialization method

        Args:
            model_path: path to the Darknet Keras model
            classes_path: path to the file with class names
            class_t: float, box score threshold for filtering
            nms_t: float, IOU threshold for non-max suppression
            anchors: numpy.ndarray with anchor box dimensions
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    
    def process_outputs(self, outputs, image_size):
    """
    Process the outputs from the model.

    Parameters:
    - outputs: list of numpy.ndarrays with shape
               (grid_h, grid_w, anchor_boxes, 4 + 1 + classes)
    - image_size: numpy.ndarray (image_height, image_width)

    Returns:
    - boxes: list of (grid_h, grid_w, anchor_boxes, 4) with (x1, y1, x2, y2)
    - box_confidences: list of (grid_h, grid_w, anchor_boxes, 1)
    - box_class_probs: list of (grid_h, grid_w, anchor_boxes, classes)
    """
    boxes = []
    box_confidences = []
    box_class_probs = []

    image_h, image_w = image_size
    input_h = self.model.input.shape[1]
    input_w = self.model.input.shape[2]

    for i, output in enumerate(outputs):
        grid_h, grid_w, anchor_boxes, _ = output.shape

        # Split output
        tx = output[..., 0]
        ty = output[..., 1]
        tw = output[..., 2]
        th = output[..., 3]

        box_confidence = tf.sigmoid(output[..., 4:5]).numpy()
        box_class_prob = tf.sigmoid(output[..., 5:]).numpy()

        # Create grid
        grid_x = np.arange(grid_w).reshape(1, grid_w, 1)
        grid_x = np.tile(grid_x, (grid_h, 1, anchor_boxes))
        grid_y = np.arange(grid_h).reshape(grid_h, 1, 1)
        grid_y = np.tile(grid_y, (1, grid_w, anchor_boxes))

        bx = (tf.sigmoid(tx).numpy() + grid_x) / grid_w
        by = (tf.sigmoid(ty).numpy() + grid_y) / grid_h

        pw = self.anchors[i, :, 0].reshape(1, 1, anchor_boxes)
        ph = self.anchors[i, :, 1].reshape(1, 1, anchor_boxes)

        bw = (np.exp(np.clip(tw, -10, 10)) * pw) / input_w
        bh = (np.exp(np.clip(th, -10, 10)) * ph) / input_h

        # Scale to image size
        x1 = (bx - bw / 2) * image_w
        y1 = (by - bh / 2) * image_h
        x2 = (bx + bw / 2) * image_w
        y2 = (by + bh / 2) * image_h

        box = np.stack([x1, y1, x2, y2], axis=-1)
        boxes.append(box)
        box_confidences.append(box_confidence)
        box_class_probs.append(box_class_prob)

    return boxes, box_confidences, box_class_probs
