#!/usr/bin/env python3
"""YOLO object detection module"""

import numpy as np


class Yolo:
    """Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = self._load_model(model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        self.class_names = self._load_classes(classes_path)

    def _load_model(self, path):
        from tensorflow.keras.models import load_model
        return load_model(path)

    def _load_classes(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape
            anchors = self.anchors[i]

            t_xy = self.sigmoid(output[..., :2])
            t_wh = output[..., 2:4]
            box_confidence = self.sigmoid(output[..., 4, np.newaxis])
            box_class_prob = self.sigmoid(output[..., 5:])

            # Create a grid of (cx, cy)
            cx = np.arange(grid_w).reshape(1, grid_w)
            cy = np.arange(grid_h).reshape(grid_h, 1)
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            grid = np.stack((cx_grid, cy_grid), axis=-1)
            grid = grid.reshape(grid_h, grid_w, 1, 2)

            # Calculate box center (bx, by)
            bx_by = (t_xy + grid) / [grid_w, grid_h]

            # Calculate box width and height (bw, bh)
            bw_bh = (np.exp(t_wh) * anchors) / [self.model.input.shape[1].value,
                                                self.model.input.shape[2].value]

            # Convert to (x1, y1, x2, y2)
            box = np.zeros(output[..., :4].shape)
            box[..., 0] = (bx_by[..., 0] - (bw_bh[..., 0] / 2)) * image_width
            box[..., 1] = (bx_by[..., 1] - (bw_bh[..., 1] / 2)) * image_height
            box[..., 2] = (bx_by[..., 0] + (bw_bh[..., 0] / 2)) * image_width
            box[..., 3] = (bx_by[..., 1] + (bw_bh[..., 1] / 2)) * image_height

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
