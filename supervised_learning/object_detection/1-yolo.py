#!/usr/bin/env python3
"""YOLO v3 Object Detection"""
import tensorflow as tf
import numpy as np


class Yolo:
    """Yolo class to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialization method"""
        self.model = tf.keras.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process model outputs to get boxes, confidences and class probabilities"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            box_confidence = tf.sigmoid(output[..., 4:5]).numpy()
            class_probs = tf.sigmoid(output[..., 5:]).numpy()

            # Create grid of cx and cy
            col = np.tile(np.arange(0, grid_w), grid_h).reshape(grid_w, grid_h).T
            row = np.tile(np.arange(0, grid_h), grid_w).reshape(grid_w, grid_h)
            cx = col[..., np.newaxis]
            cy = row[..., np.newaxis]

            # Apply sigmoid to t_xy and compute bx, by
            bx = (tf.sigmoid(t_xy[..., 0]) + cx) / grid_w
            by = (tf.sigmoid(t_xy[..., 1]) + cy) / grid_h

            # Compute bw, bh using anchors and input model shape
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            bw = (np.exp(t_wh[..., 0]) * pw) / self.model.input.shape[1].value
            bh = (np.exp(t_wh[..., 1]) * ph) / self.model.input.shape[2].value

            # Calculate corners
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
