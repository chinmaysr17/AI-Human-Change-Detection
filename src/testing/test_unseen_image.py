"""
Test Improved CNN on an Unseen LEVIR-CD Image Pair
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to Python path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import config

from data.image_loader import load_image_pair
from preprocessing.image_loader import preprocess_images
from tensorflow.keras.models import load_model


def main():

    print("=" * 60)
    print("UNSEEN IMAGE CHANGE DETECTION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Select image pair
    # --------------------------------------------------

    filename = "test_2.png"

    print("\nLoading unseen LEVIR-CD image pair...")
    print("Image pair:", filename)

    image_a, image_b, label = load_image_pair(filename)

    if image_a is None:

        print("\n❌ Unable to load image pair.")
        return

    print("✅ Image A loaded.")
    print("✅ Image B loaded.")
    print("✅ Ground Truth loaded.")

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

    print(
        "Image A shape:",
        processed_a.shape
    )

    print(
        "Image B shape:",
        processed_b.shape
    )

    # --------------------------------------------------
    # 3. Create 6-channel input
    # --------------------------------------------------

    combined_image = np.concatenate(
        [processed_a, processed_b],
        axis=-1
    )

    model_input = np.expand_dims(
        combined_image,
        axis=0
    )

    print(
        "\nModel input shape:",
        model_input.shape
    )

    # --------------------------------------------------
    # 4. Load improved CNN
    # --------------------------------------------------

    model_path = os.path.join(
        config.BASE_DIR,
        "models",
        "change_detection_cnn_improved.keras"
    )

    print("\nLoading improved CNN...")
    print("Model path:", model_path)

    if not os.path.exists(model_path):

        print("\n❌ Improved model not found.")
        return

    model = load_model(
        model_path,
        compile=False
    )

    print("✅ Improved CNN loaded.")

    # --------------------------------------------------
    # 5. Generate prediction
    # --------------------------------------------------

    print("\nGenerating prediction...")

    prediction = model.predict(
        model_input,
        verbose=0
    )[0]

    print("✅ Prediction generated.")

    print(
        "Prediction shape:",
        prediction.shape
    )

    print(
        "Prediction range:",
        prediction.min(),
        "to",
        prediction.max()
    )

    # --------------------------------------------------
    # 6. Convert prediction to binary map
    # --------------------------------------------------

    threshold = 0.50

    predicted_change = (
        prediction >= threshold
    ).astype(np.uint8)

    predicted_change = predicted_change.squeeze()

    print(
        "\nThreshold:",
        threshold
    )

    print(
        "Predicted change pixels:",
        np.sum(predicted_change == 1)
    )

    # --------------------------------------------------
    # 7. Prepare ground truth
    # --------------------------------------------------

    ground_truth = processed_label.astype(
        np.uint8
    )

    # --------------------------------------------------
    # 8. Create overlay
    # --------------------------------------------------

    overlay = processed_b.copy()

    change_mask = predicted_change == 1

    # Highlight detected changes
    overlay[change_mask] = [
        1.0,
        0.0,
        0.0
    ]

    overlay = (
        0.5 * processed_b
        + 0.5 * overlay
    )

    # --------------------------------------------------
    # 9. Display results
    # --------------------------------------------------

    print("\nCreating visualization...")

    plt.figure(
        figsize=(20, 4)
    )

    # Before image
    plt.subplot(1, 5, 1)

    plt.imshow(
        processed_a
    )

    plt.title(
        "Before Image (A)"
    )

    plt.axis("off")

    # After image
    plt.subplot(1, 5, 2)

    plt.imshow(
        processed_b
    )

    plt.title(
        "After Image (B)"
    )

    plt.axis("off")

    # Ground truth
    plt.subplot(1, 5, 3)

    plt.imshow(
        ground_truth,
        cmap="gray"
    )

    plt.title(
        "Ground Truth"
    )

    plt.axis("off")

    # Prediction
    plt.subplot(1, 5, 4)

    plt.imshow(
        predicted_change,
        cmap="gray"
    )

    plt.title(
        "Predicted Change"
    )

    plt.axis("off")

    # Overlay
    plt.subplot(1, 5, 5)

    plt.imshow(
        overlay
    )

    plt.title(
        "Change Overlay"
    )

    plt.axis("off")

    plt.tight_layout()

    # --------------------------------------------------
    # 10. Save result
    # --------------------------------------------------

    output_dir = os.path.join(
        config.BASE_DIR,
        "outputs"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        "unseen_image_test_result.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\n✅ Unseen image result saved."
    )

    print(
        "Output path:",
        output_path
    )

    plt.show()

    print(
        "\n✅ Unseen image testing completed successfully."
    )


if __name__ == "__main__":
    main()