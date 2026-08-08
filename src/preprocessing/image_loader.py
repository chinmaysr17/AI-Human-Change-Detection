"""
Image Preprocessing Module
"""

import cv2
import numpy as np
import config


def preprocess_images(image_a, image_b, label):
    """
    Resize and normalize satellite images.
    Prepare the ground-truth change mask.
    """

    # Resize satellite images
    image_a = cv2.resize(
        image_a,
        config.IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    image_b = cv2.resize(
        image_b,
        config.IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    # Resize label using nearest-neighbor interpolation
    label = cv2.resize(
        label,
        config.IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    # Convert BGR to RGB
    image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
    image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)

    # Normalize images from 0-255 to 0-1
    image_a = image_a.astype(np.float32) / 255.0
    image_b = image_b.astype(np.float32) / 255.0

    # Convert label into binary mask
    label = (label > 0).astype(np.uint8)

    return image_a, image_b, label