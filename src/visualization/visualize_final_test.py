"""
Final Test Set Visualization

Visualizes one sample from the completely held-out
13-image test set using the final CNN model.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to Python path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import config

from training.data_loader import load_dataset
from training.split_train_val_test import split_train_val_test
from tensorflow.keras.models import load_model


def main():

    print("=" * 60)
    print("FINAL TEST SET VISUALIZATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load complete dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    images, labels = load_dataset()

    print("✅ Dataset loaded.")

    # --------------------------------------------------
    # 2. Recreate the exact dataset split
    # --------------------------------------------------

    print("\nCreating the same train/validation/test split...")

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

    print("✅ Test set recreated.")

    print(
        "Final test samples:",
        len(X_test)
    )

    # --------------------------------------------------
    # 3. Select one held-out test sample
    # --------------------------------------------------

    test_index = 0

    test_image = X_test[test_index]
    test_label = y_test[test_index]

    print(
        "\nVisualizing final test sample:",
        test_index + 1
    )

    print(
        "Test image shape:",
        test_image.shape
    )

    print(
        "Test label shape:",
        test_label.shape
    )

    # --------------------------------------------------
    # 4. Separate Image A and Image B
    # --------------------------------------------------

    image_a = test_image[:, :, :3]

    image_b = test_image[:, :, 3:6]

    ground_truth = test_label.squeeze()

    # --------------------------------------------------
    # 5. Load final trained model
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

        print("\n❌ Final model not found.")

        return

    model = load_model(
        model_path,
        compile=False
    )

    print("✅ Final CNN loaded.")

    # --------------------------------------------------
    # 6. Generate prediction
    # --------------------------------------------------

    print("\nGenerating prediction...")

    model_input = np.expand_dims(
        test_image,
        axis=0
    )

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
    # 7. Create binary prediction
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
        np.sum(
            predicted_change == 1
        )
    )

    print(
        "Ground-truth change pixels:",
        np.sum(
            ground_truth == 1
        )
    )

    # --------------------------------------------------
    # 8. Create overlay
    # --------------------------------------------------

    overlay = image_b.copy()

    change_mask = predicted_change == 1

    overlay[change_mask] = [
        1.0,
        0.0,
        0.0
    ]

    overlay = (
        0.5 * image_b
        + 0.5 * overlay
    )

    # --------------------------------------------------
    # 9. Create visualization
    # --------------------------------------------------

    print("\nCreating final visualization...")

    plt.figure(
        figsize=(20, 4)
    )

    # Before image
    plt.subplot(1, 5, 1)

    plt.imshow(image_a)

    plt.title(
        "Before Image (A)"
    )

    plt.axis("off")

    # After image
    plt.subplot(1, 5, 2)

    plt.imshow(image_b)

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
        "final_test_visualization.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\n✅ Final test visualization saved."
    )

    print(
        "Output path:",
        output_path
    )

    # Display result
    plt.show()

    print(
        "\n✅ Final test visualization completed successfully."
    )


if __name__ == "__main__":
    main()