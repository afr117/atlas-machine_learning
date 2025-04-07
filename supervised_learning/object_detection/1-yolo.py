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
        """Process model outputs to get boxes,
        confidences and class probabilities"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            box_confidence = tf.sigmoid(output[..., 4:5]).numpy()
            class_probs = tf.sigmoid(output[..., 5:]).numpy()

            # Grid offsets
            cx = np.arange(grid_w).reshape(1, grid_w)
            cy = np.arange(grid_h).reshape(grid_h, 1)
            cx = np.tile(cx, (grid_h, 1))
            cy = np.tile(cy, (1, grid_w))

            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            tx = tf.sigmoid(t_xy[..., 0]) + cx
            ty = tf.sigmoid(t_xy[..., 1]) + cy

            tx /= grid_w
            ty /= grid_h

            tw = t_wh[..., 0]
            th = t_wh[..., 1]

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            pw = pw.reshape(1, 1, anchor_boxes)
            ph = ph.reshape(1, 1, anchor_boxes)

            tw = np.exp(tw) * pw / input_w
            th = np.exp(th) * ph / input_h

            x1 = (tx - tw / 2) * image_width
            y1 = (ty - th / 2) * image_height
            x2 = (tx + tw / 2) * image_width
            y2 = (ty + th / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
