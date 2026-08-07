"""
Image Loader Module
Loads satellite images using OpenCV.
"""

import cv2


def load_image(image_path):
    """
    Load an image from the given path.

    Args:
        image_path (str): Path to the image.

    Returns:
        image: Loaded image or None if loading fails.
    """

    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error: Unable to load image: {image_path}")
        return None

    print("✅ Image loaded successfully.")
    return image