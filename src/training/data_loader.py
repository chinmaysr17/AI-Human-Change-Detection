"""
Training Data Loader
"""

import os
import sys
import numpy as np

# Add src directory to Python path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import config
from data.image_loader import load_image_pair
from preprocessing.image_loader import preprocess_images


def load_dataset():

    print("\nLoading LEVIR-CD training dataset...\n")

    image_files = sorted(
        [
            file
            for file in os.listdir(config.IMAGE_A_DIR)
            if file.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    images = []
    labels = []

    for index, filename in enumerate(image_files):

        print(
            f"Loading image {index + 1}/{len(image_files)}: {filename}"
        )

        # Load A, B and label
        image_a, image_b, label = load_image_pair(filename)

        if image_a is None:
            print(f"❌ Skipping {filename}")
            continue

        # Preprocess
        processed_a, processed_b, processed_label = preprocess_images(
            image_a,
            image_b,
            label
        )

        # Combine Image A and Image B
        combined_image = np.concatenate(
            [processed_a, processed_b],
            axis=-1
        )

        # Add channel dimension to label
        processed_label = np.expand_dims(
            processed_label,
            axis=-1
        )

        images.append(combined_image)
        labels.append(processed_label)

    # Convert lists to NumPy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    print("\n========================================")
    print("Dataset Loading Completed")
    print("========================================")

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    return images, labels