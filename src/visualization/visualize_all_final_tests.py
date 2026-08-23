"""
Visualize All 13 Final Test Images

Uses the completely held-out test set and the final CNN model.

For every test image, saves:
    1. Before Image
    2. After Image
    3. Ground Truth
    4. Final CNN Prediction
    5. Change Overlay
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Add src directory to Python path
# --------------------------------------------------

SRC_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, SRC_DIR)

import config

from training.data_loader import load_dataset
from training.split_train_val_test import split_train_val_test
from tensorflow.keras.models import load_model


def create_overlay(image, prediction):

    """
    Highlight predicted changes on the After image.
    """

    overlay = image.copy()

    change_mask = prediction == 1

    overlay[change_mask] = [
        1.0,
        0.0,
        0.0
    ]

    overlay = (
        0.5 * image
        + 0.5 * overlay
    )

    return overlay


def main():

    print("=" * 60)
    print("ALL FINAL TEST SET VISUALIZATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    images, labels = load_dataset()

    print("✅ Dataset loaded.")

    print(
        "Images shape:",
        images.shape
    )

    print(
        "Labels shape:",
        labels.shape
    )

    # --------------------------------------------------
    # 2. Recreate exact test split
    # --------------------------------------------------

    print("\nCreating test split...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_train_val_test(
        images,
        labels,
        test_size=0.10,
        validation_size=0.10,
        random_state=42
    )

    print("✅ Test split recreated.")

    print(
        "Final test samples:",
        len(X_test)
    )

    # --------------------------------------------------
    # 3. Load final model
    # --------------------------------------------------

    model_path = os.path.join(
        config.BASE_DIR,
        "models",
        "change_detection_cnn_final.keras"
    )

    print("\nLoading final CNN model...")

    print(
        "Model path:",
        model_path
    )

    if not os.path.exists(model_path):

        print("\n❌ Final CNN model not found.")

        return

    model = load_model(
        model_path,
        compile=False
    )

    print("✅ Final CNN model loaded.")

    # --------------------------------------------------
    # 4. Create output directory
    # --------------------------------------------------

    output_dir = os.path.join(
        config.BASE_DIR,
        "outputs",
        "final_test_visualizations"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print(
        "\nOutput directory:",
        output_dir
    )

    # --------------------------------------------------
    # 5. Prediction threshold
    # --------------------------------------------------

    threshold = 0.50

    print(
        "\nPrediction threshold:",
        threshold
    )

    # --------------------------------------------------
    # 6. Process every test image
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    for index in range(len(X_test)):

        print(
            f"\nProcessing test image "
            f"{index + 1}/{len(X_test)}..."
        )

        # ----------------------------------------------
        # Get test image and label
        # ----------------------------------------------

        test_image = X_test[index]

        test_label = y_test[index]

        # ----------------------------------------------
        # Separate Before and After images
        # ----------------------------------------------

        image_a = test_image[:, :, :3]

        image_b = test_image[:, :, 3:6]

        ground_truth = test_label.squeeze()

        # ----------------------------------------------
        # Generate prediction
        # ----------------------------------------------

        model_input = np.expand_dims(
            test_image,
            axis=0
        )

        prediction = model.predict(
            model_input,
            verbose=0
        )[0]

        # ----------------------------------------------
        # Convert probability to binary map
        # ----------------------------------------------

        predicted_change = (
            prediction >= threshold
        ).astype(np.uint8)

        predicted_change = (
            predicted_change.squeeze()
        )

        # ----------------------------------------------
        # Create overlay
        # ----------------------------------------------

        overlay = create_overlay(
            image_b,
            predicted_change
        )

        # ----------------------------------------------
        # Calculate pixel counts
        # ----------------------------------------------

        predicted_pixels = np.sum(
            predicted_change == 1
        )

        ground_truth_pixels = np.sum(
            ground_truth == 1
        )

        print(
            "Predicted change pixels:",
            predicted_pixels
        )

        print(
            "Ground truth change pixels:",
            ground_truth_pixels
        )

        # ----------------------------------------------
        # Create figure
        # ----------------------------------------------

        plt.figure(
            figsize=(20, 4)
        )

        # Before Image
        plt.subplot(1, 5, 1)

        plt.imshow(
            image_a
        )

        plt.title(
            "Before Image (A)"
        )

        plt.axis("off")

        # After Image
        plt.subplot(1, 5, 2)

        plt.imshow(
            image_b
        )

        plt.title(
            "After Image (B)"
        )

        plt.axis("off")

        # Ground Truth
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
            "Final CNN Prediction"
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

        # ----------------------------------------------
        # Save visualization
        # ----------------------------------------------

        output_path = os.path.join(
            output_dir,
            f"test_image_{index + 1:02d}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "✅ Saved:",
            output_path
        )

    # --------------------------------------------------
    # 7. Completion message
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("ALL TEST VISUALIZATIONS COMPLETED")
    print("=" * 60)

    print(
        "\nTotal visualizations:",
        len(X_test)
    )

    print(
        "\nSaved location:"
    )

    print(
        output_dir
    )

    print(
        "\n✅ All 13 final test visualizations saved successfully."
    )


if __name__ == "__main__":
    main()