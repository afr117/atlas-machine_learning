#!/usr/bin/env python3
"""YOLO object detection module"""

import numpy as np
import cv2
import os


class Yolo:
    """Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor

        Args:
            model_path (str): path to where a Darknet Keras model is stored
            classes_path (str): path to where the list of class names is stored
            class_t (float): box score threshold for the initial filtering step
            nms_t (float): IOU threshold for non-max suppression
            anchors (numpy.ndarray): all anchor boxes
        """
        self.model = self._load_model(model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        self.class_names = self._load_classes(classes_path)

    def _load_model(self, path):
        """Loads a pretrained Keras model from the specified path"""
        from tensorflow.keras.models import load_model
        return load_model(path)

    def _load_classes(self, path):
        """Reads class names from file, one class per line"""
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def sigmoid(self, x):
        """Applies the sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Processes the output predictions from the model

        Args:
            outputs (list of np.ndarray): contains predictions from the model
            image_size (np.ndarray): original size of the input image (h, w)

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
            - boxes (list): processed boundary boxes as (x1, y1, x2, y2)
            - box_confidences (list): probability of object presence
            - box_class_probs (list): probabilities for each class
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape
            anchors = self.anchors[i]

            t_xy = self.sigmoid(output[..., :2])
            t_wh = output[..., 2:4]
            box_confidence = self.sigmoid(output[..., 4, np.newaxis])
            box_class_prob = self.sigmoid(output[..., 5:])

            # Create grid of (cx, cy)
            cx = np.arange(grid_w).reshape(1, grid_w)
            cy = np.arange(grid_h).reshape(grid_h, 1)
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            grid = np.stack((cx_grid, cy_grid), axis=-1)
            grid = grid.reshape(grid_h, grid_w, 1, 2)

            # Box center
            bx_by = (t_xy + grid) / [grid_w, grid_h]

            # Corrected: anchors normalized with [input_h, input_w]
            bw_bh = (np.exp(t_wh) * anchors) / [input_h, input_w]

            # Convert to corner box format (x1, y1, x2, y2)
            box = np.zeros(output[..., :4].shape)
            box[..., 0] = (bx_by[..., 0] - (bw_bh[..., 0] / 2)) * image_width
            box[..., 1] = (bx_by[..., 1] - (bw_bh[..., 1] / 2)) * image_height
            box[..., 2] = (bx_by[..., 0] + (bw_bh[..., 0] / 2)) * image_width
            box[..., 3] = (bx_by[..., 1] + (bw_bh[..., 1] / 2)) * image_height

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def preprocess_images(self, images):
        """Resizes and normalizes a list of images for the YOLO model

        Args:
            images (list of np.ndarray): list of original images

        Returns:
            tuple: (pimages, image_shapes)
                - pimages: np.ndarray of shape (ni, input_h, input_w, 3)
                - image_shapes: np.ndarray of shape (ni, 2) with original (h, w)
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]
        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append(img.shape[:2])
            resized = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            normalized = resized.astype(np.float32) / 255.0
            pimages.append(normalized)

        return np.array(pimages), np.array(image_shapes)

    def load_images(self, folder_path):
        """Loads all images from a folder

        Args:
            folder_path (str): directory path to load images from

        Returns:
            tuple: (images, image_paths)
                - images: list of np.ndarray images
                - image_paths: list of image file paths
        """
        images = []
        image_paths = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, file_name)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    image_paths.append(img_path)

        return images, image_paths
