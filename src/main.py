"""
Main Entry Point
"""

import os
import config
from preprocessing.image_loader import load_image


def main():

    print("=" * 60)
    print(config.PROJECT_NAME)
    print("=" * 60)

    print("Dataset Directory")
    print(config.DATASET_DIR)
    print("=" * 60)

    image_path = os.path.join(config.DATASET_DIR, "sample.jpg")

    image = load_image(image_path)

    if image is None:
        print("\nWaiting for dataset...")
    else:
        print("Image Ready for Preprocessing")


if __name__ == "__main__":
    main()