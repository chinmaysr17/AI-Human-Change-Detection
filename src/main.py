"""
Main Entry Point
"""

import numpy as np
import config

from data.dataset_validator import validate_dataset
from data.image_loader import load_image_pair
from preprocessing.image_loader import preprocess_images
from visualization.display_images import display_image_pair


def main():

    print("=" * 60)
    print(config.PROJECT_NAME)
    print("=" * 60)

    # Validate dataset
    if not validate_dataset():
        print("\n❌ Dataset validation failed.")
        return

    # Load first image pair
    print("\nLoading first image pair...\n")

    image_a, image_b, label = load_image_pair("test_1.png")

    if image_a is None:
        print("\n❌ Failed to load image pair.")
        return

    print("\n✅ First LEVIR-CD image pair loaded successfully.")

    print("Image A shape:", image_a.shape)
    print("Image B shape:", image_b.shape)
    print("Label shape  :", label.shape)

    # Preprocess images
    print("\nPreprocessing images...")

    processed_a, processed_b, processed_label = preprocess_images(
        image_a,
        image_b,
        label
    )

    print("✅ Preprocessing completed.")

    print("Processed Image A shape:", processed_a.shape)
    print("Processed Image B shape:", processed_b.shape)
    print("Processed Label shape  :", processed_label.shape)

    print(
        "Image A pixel range:",
        processed_a.min(),
        "to",
        processed_a.max()
    )

    print(
        "Image B pixel range:",
        processed_b.min(),
        "to",
        processed_b.max()
    )

    print("Label values:", np.unique(processed_label))

    # Display preprocessed images
    print("\nDisplaying preprocessed images...")

    display_image_pair(
        (processed_a * 255).astype(np.uint8),
        (processed_b * 255).astype(np.uint8),
        processed_label
    )


if __name__ == "__main__":
    main()
