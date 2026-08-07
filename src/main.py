"""
AI-Based Human-Induced Change Detection Using Satellite Images
Main Entry Point
"""

import config
from preprocessing.image_loader import load_image


def main():

    print("=" * 60)
    print(config.PROJECT_NAME)
    print("=" * 60)

    image_path = "dataset/sample.jpg"

    image = load_image(image_path)

    if image is None:
        print("Please add a satellite image named 'sample.jpg' inside the dataset folder.")
    else:
        print("Image Ready for Preprocessing.")


if __name__ == "__main__":
    main()