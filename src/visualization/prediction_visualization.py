"""
CNN Prediction Visualization
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
    print("CNN PREDICTION VISUALIZATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load LEVIR-CD sample
    # --------------------------------------------------

    filename = "test_1.png"

    print("\nLoading LEVIR-CD sample...")
    print("Sample:", filename)

    image_a, image_b, label = load_image_pair(filename)

    if image_a is None:

        print("\n❌ Unable to load sample.")
        return

    print("✅ Image A loaded.")
    print("✅ Image B loaded.")
    print("✅ Label loaded.")

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

    # --------------------------------------------------
    # 3. Combine Image A and Image B
    # --------------------------------------------------

    combined_image = np.concatenate(
        [processed_a, processed_b],
        axis=-1
    )

    model_input = np.expand_dims(
        combined_image,
        axis=0
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

    print("\nGenerating change prediction...")

    prediction = model.predict(
        model_input,
        verbose=0
    )[0]

    print("✅ Prediction generated.")

    print(
        "Prediction range:",
        prediction.min(),
        "to",
        prediction.max()
    )

    # --------------------------------------------------
    # 6. Convert prediction to binary change map
    # --------------------------------------------------

    threshold = 0.50

    predicted_change = (
        prediction >= threshold
    ).astype(np.uint8)

    predicted_change = predicted_change.squeeze()

    print(
        "\nPrediction threshold:",
        threshold
    )

    print(
        "Predicted change pixels:",
        np.sum(predicted_change == 1)
    )

    # --------------------------------------------------
    # 7. Prepare ground truth
    # --------------------------------------------------

    ground_truth = processed_label.astype(np.uint8)

    # --------------------------------------------------
    # 8. Create change overlay
    # --------------------------------------------------

    # Start with Image B
    overlay = processed_b.copy()

    # Highlight predicted changes
    change_mask = predicted_change == 1

    # Use a bright red highlight for detected changes
    overlay[change_mask] = [1.0, 0.0, 0.0]

    # Blend original Image B and predicted change overlay
    overlay = (
        0.5 * processed_b
        + 0.5 * overlay
    )

    # --------------------------------------------------
    # 9. Create visualization
    # --------------------------------------------------

    print("\nCreating visualization...")

    plt.figure(figsize=(20, 4))

    # --------------------------------------------------
    # Image A
    # --------------------------------------------------

    plt.subplot(1, 5, 1)

    plt.imshow(processed_a)

    plt.title("Before Image (A)")

    plt.axis("off")

    # --------------------------------------------------
    # Image B
    # --------------------------------------------------

    plt.subplot(1, 5, 2)

    plt.imshow(processed_b)

    plt.title("After Image (B)")

    plt.axis("off")

    # --------------------------------------------------
    # Ground Truth
    # --------------------------------------------------

    plt.subplot(1, 5, 3)

    plt.imshow(
        ground_truth,
        cmap="gray"
    )

    plt.title("Ground Truth")

    plt.axis("off")

    # --------------------------------------------------
    # Predicted Change
    # --------------------------------------------------

    plt.subplot(1, 5, 4)

    plt.imshow(
        predicted_change,
        cmap="gray"
    )

    plt.title(
        "Predicted Change\n"
        f"Threshold = {threshold}"
    )

    plt.axis("off")

    # --------------------------------------------------
    # Overlay
    # --------------------------------------------------

    plt.subplot(1, 5, 5)

    plt.imshow(overlay)

    plt.title(
        "Change Overlay"
    )

    plt.axis("off")

    plt.tight_layout()

    # --------------------------------------------------
    # 10. Save visualization
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
        "change_detection_result_overlay.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\n✅ Visualization saved successfully."
    )

    print(
        "Output path:",
        output_path
    )

    # Display result
    plt.show()


if __name__ == "__main__":
    main()