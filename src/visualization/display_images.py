"""
Visualization Module
"""

import cv2
import matplotlib.pyplot as plt


def display_image_pair(image_a, image_b, label):
    """
    Display before image, after image, and ground-truth label.
    """

    # Convert OpenCV BGR images to RGB
    image_a_rgb = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
    image_b_rgb = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Before image
    plt.subplot(1, 3, 1)
    plt.imshow(image_a_rgb)
    plt.title("Before Image (A)")
    plt.axis("off")

    # After image
    plt.subplot(1, 3, 2)
    plt.imshow(image_b_rgb)
    plt.title("After Image (B)")
    plt.axis("off")

    # Ground truth
    plt.subplot(1, 3, 3)
    plt.imshow(label, cmap="gray")
    plt.title("Ground Truth Label")
    plt.axis("off")

    plt.tight_layout()
    plt.show()