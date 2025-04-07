#!/usr/bin/env python3
"""YOLO v3 Object Detection - Process Outputs"""

import tensorflow as tf
import numpy as np


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor

        Args:
            model_path (str): path to a Darknet Keras model
            classes_path (str): path to class names file
            class_t (float): box score threshold for initial filtering
            nms_t (float): IOU threshold for non-max suppression
            anchors (np.ndarray): anchor boxes (outputs, anchor_boxes, 2)
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process model outputs for a single image

        Args:
            outputs (list): list of numpy.ndarrays with predictions
            image_size (np.ndarray): original image size [height, width]

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h, input_w = self.model.input.shape[1:3].as_list()
        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            box = output[..., :4]
            box_confidence = self.sigmoid(output[..., 4:5])
            class_probs = self.sigmoid(output[..., 5:])

            anchors = self.anchors[i]

            # Create grid of shape (grid_h, grid_w, 1)
            cy = np.arange(grid_h).reshape(-1, 1, 1)
            cx = np.arange(grid_w).reshape(1, -1, 1)
            cy = np.tile(cy, (1, grid_w, anchor_boxes))
            cx = np.tile(cx, (grid_h, 1, anchor_boxes))

            # Offset for each box
            t_x = box[..., 0]
            t_y = box[..., 1]
            t_w = box[..., 2]
            t_h = box[..., 3]

            bx = self.sigmoid(t_x) + cx
            by = self.sigmoid(t_y) + cy
            bw = anchors[:, 0] * np.exp(t_w)
            bh = anchors[:, 1] * np.exp(t_h)

            # Normalize to original image scale
            bx /= grid_w
            by /= grid_h
            bw /= input_w
            bh /= input_h

            # Calculate corners
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box_xy = np.stack((x1, y1, x2, y2), axis=-1)

            boxes.append(box_xy)
            box_confidences.append(box_confidence)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
