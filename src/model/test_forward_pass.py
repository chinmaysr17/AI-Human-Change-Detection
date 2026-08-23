"""
Test CNN Forward Pass
"""

import os
import sys
import numpy as np

# Add the src directory to Python's import path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from data.image_loader import load_image_pair
from preprocessing.image_loader import preprocess_images
from model.change_detection_cnn import build_change_detection_model


def main():

    print("=" * 60)
    print("CNN FORWARD PASS TEST")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load LEVIR-CD image pair
    # --------------------------------------------------

    print("\nLoading LEVIR-CD sample...")

    image_a, image_b, label = load_image_pair("test_1.png")

    if image_a is None:
        print("❌ Failed to load image pair.")
        return

    print("✅ Image pair loaded.")

    # --------------------------------------------------
    # 2. Preprocess images
    # --------------------------------------------------

    print("\nPreprocessing images...")

    processed_a, processed_b, processed_label = preprocess_images(
        image_a,
        image_b,
        label
    )

    print("✅ Preprocessing completed.")

    print("Image A:", processed_a.shape)
    print("Image B:", processed_b.shape)
    print("Label  :", processed_label.shape)

    # --------------------------------------------------
    # 3. Combine Image A and Image B
    # --------------------------------------------------

    combined_input = np.concatenate(
        [processed_a, processed_b],
        axis=-1
    )

    print("\nCombined input shape:", combined_input.shape)

    # --------------------------------------------------
    # 4. Add batch dimension
    # --------------------------------------------------

    model_input = np.expand_dims(
        combined_input,
        axis=0
    )

    print("Model input shape:", model_input.shape)

    # --------------------------------------------------
    # 5. Build CNN
    # --------------------------------------------------

    print("\nBuilding CNN model...")

    model = build_change_detection_model()

    print("✅ CNN model created.")

    # --------------------------------------------------
    # 6. Run prediction
    # --------------------------------------------------

    print("\nRunning CNN prediction...")

    prediction = model.predict(
        model_input,
        verbose=0
    )

    print("✅ Prediction completed.")

    # --------------------------------------------------
    # 7. Check prediction output
    # --------------------------------------------------

    print("\nPrediction shape:", prediction.shape)

    print(
        "Prediction range:",
        prediction.min(),
        "to",
        prediction.max()
    )

    # --------------------------------------------------
    # 8. Final verification
    # --------------------------------------------------

    expected_shape = (1, 256, 256, 1)

    if prediction.shape == expected_shape:

        print("\n✅ Prediction output shape is correct.")
        print("✅ CNN forward pass successful.")

    else:

        print("\n❌ Unexpected prediction shape.")
        print("Expected:", expected_shape)
        print("Received:", prediction.shape)


if __name__ == "__main__":
    main()