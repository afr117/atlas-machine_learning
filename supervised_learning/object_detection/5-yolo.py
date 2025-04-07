#!/usr/bin/env python3
"""YOLO v3 Object Detection"""
import tensorflow as tf
import numpy as np
import os
import cv2


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

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            box_confidence = tf.sigmoid(output[..., 4:5]).numpy()
            class_probs = tf.sigmoid(output[..., 5:]).numpy()

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

            pw = self.anchors[i, :, 0].reshape(1, 1, anchor_boxes)
            ph = self.anchors[i, :, 1].reshape(1, 1, anchor_boxes)

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with box scores above threshold"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_score = box_confidences[i] * box_class_probs[i]
            class_ids = np.argmax(box_score, axis=-1)
            class_scores = np.max(box_score, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(class_ids[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply Non-Max Suppression to filter overlapping boxes"""
        final_boxes = []
        final_classes = []
        final_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)
            boxes = filtered_boxes[idxs]
            scores = box_scores[idxs]

            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort()[::-1]

            while order.size > 0:
                i = order[0]
                final_boxes.append(boxes[i])
                final_classes.append(cls)
                final_scores.append(scores[i])

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(ovr <= self.nms_t)[0]
                order = order[inds + 1]

        return (np.array(final_boxes),
                np.array(final_classes),
                np.array(final_scores))

    @staticmethod
    def load_images(folder_path):
        """Loads all images from a given folder"""
        image_paths = []
        images = []

        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            if os.path.isfile(path):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                    image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess images: resize and normalize"""
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_shapes = np.array([img.shape[:2] for img in images])
        pimages = []

        for img in images:
            resized = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            normalized = resized.astype(np.float32) / 255.0
            pimages.append(normalized)

        pimages = np.stack(pimages, axis=0)
        return pimages, image_shapes
