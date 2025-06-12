import json

import cv2
import numpy as np


def load_calibration_json(filename):
    with open(filename) as f:
        data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"])
    dist_coeff = np.array(data["dist_coeff"])
    image_size = tuple(data["image_size"])

    return camera_matrix, dist_coeff, image_size


def load_rgb(path):
    """
    Load image from path and convert to RGB.
    """
    return cv2.imread(path, cv2.IMREAD_COLOR_RGB)
