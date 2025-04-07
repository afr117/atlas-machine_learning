#!/usr/bin/env python3
"""YOLO v3 Object Detection - Process Outputs"""

import tensorflow as tf
import numpy as np


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the YOLO object detector

        Args:
            model_path (str): path to a Darknet Keras model file (.h5)
            classes_path (str): path to class names file
            class_t (float): box score threshold
            nms_t (float): IOU threshold
            anchors (np.ndarray): anchor boxes (outputs, anchor_boxes, 2)
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process YOLO outputs for one image

        Args:
            outputs: List of numpy.ndarrays with shape (gh, gw, anchors, 85)
            image_size: numpy.ndarray [height, width]

        Returns:
            boxes, box_confidences, box_class_probs
        """
        image_h, image_w = image_size
        input_h, input_w = self.model.input.shape[1:3].as_list()

        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            gh, gw, num_anchors, _ = output.shape

            # Extract t_x, t_y, t_w, t_h
            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            box_conf = self.sigmoid(output[..., 4:5])
            class_probs = self.sigmoid(output[..., 5:])

            # Generate grid for center offsets
            grid_y = np.arange(gh).reshape(-1, 1, 1)
            grid_x = np.arange(gw).reshape(1, -1, 1)
            cx = np.tile(grid_x, (gh, 1, num_anchors))
            cy = np.tile(grid_y, (1, gw, num_anchors))

            # Shape anchors for broadcasting
            anchor_w = self.anchors[i][:, 0].reshape((1, 1, num_anchors))
            anchor_h = self.anchors[i][:, 1].reshape((1, 1, num_anchors))

            # Compute center coordinates
            bx = (self.sigmoid(t_xy[..., 0]) + cx) / gw
            by = (self.sigmoid(t_xy[..., 1]) + cy) / gh

            # Compute box dimensions
            bw = (np.exp(t_wh[..., 0]) * anchor_w) / input_w
            bh = (np.exp(t_wh[..., 1]) * anchor_h) / input_h

            # Convert center coords to corners (x1, y1, x2, y2)
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)
            box_confidences.append(box_conf)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
