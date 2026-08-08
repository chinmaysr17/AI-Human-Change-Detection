"""
LEVIR-CD Image Loader
"""

import cv2
import config


def load_image_pair(filename):
    """
    Load before image, after image, and ground-truth label.
    """

    image_a_path = config.IMAGE_A_DIR + "\\" + filename
    image_b_path = config.IMAGE_B_DIR + "\\" + filename
    label_path = config.LABEL_DIR + "\\" + filename

    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    if image_a is None:
        print(f"❌ Unable to load Image A: {image_a_path}")
        return None, None, None

    if image_b is None:
        print(f"❌ Unable to load Image B: {image_b_path}")
        return None, None, None

    if label is None:
        print(f"❌ Unable to load Label: {label_path}")
        return None, None, None

    print("✅ Image A loaded successfully.")
    print("✅ Image B loaded successfully.")
    print("✅ Label loaded successfully.")

    return image_a, image_b, label