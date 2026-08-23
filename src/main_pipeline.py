"""
End-to-End Human-Induced Change Detection Pipeline

Flow:
Image Pair
    ↓
Image Loading
    ↓
Preprocessing
    ↓
6-Channel Input
    ↓
Final CNN
    ↓
Change Prediction
    ↓
Thresholding
    ↓
Change Map
    ↓
Visualization
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. ADD SRC DIRECTORY TO PYTHON PATH
# ============================================================

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(0, SRC_DIR)


# ============================================================
# 2. IMPORT PROJECT MODULES
# ============================================================

import config

from data.image_loader import load_image_pair
from preprocessing.image_loader import preprocess_images

from tensorflow.keras.models import load_model


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("AI-BASED HUMAN-INDUCED CHANGE DETECTION")
    print("END-TO-END PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: SELECT IMAGE PAIR
    # --------------------------------------------------------

    filename = "test_2.png"

    print("\n[1/7] Loading satellite image pair...")
    print("Image pair:", filename)

    image_a, image_b, label = load_image_pair(
        filename
    )

    if image_a is None or image_b is None:

        print(
            "\n❌ Image pair could not be loaded."
        )

        return

    print("✅ Image A loaded.")
    print("✅ Image B loaded.")

    if label is not None:

        print("✅ Ground truth label loaded.")

    else:

        print(
            "⚠️ Ground truth label not available."
        )


    # --------------------------------------------------------
    # STEP 2: PREPROCESS IMAGES
    # --------------------------------------------------------

    print("\n[2/7] Preprocessing images...")

    processed_a, processed_b, processed_label = (
        preprocess_images(
            image_a,
            image_b,
            label
        )
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


    # --------------------------------------------------------
    # STEP 3: CREATE 6-CHANNEL INPUT
    # --------------------------------------------------------

    print(
        "\n[3/7] Creating 6-channel model input..."
    )

    combined_image = np.concatenate(
        [
            processed_a,
            processed_b
        ],
        axis=-1
    )

    print(
        "Combined input shape:",
        combined_image.shape
    )

    # Add batch dimension

    model_input = np.expand_dims(
        combined_image,
        axis=0
    )

    print(
        "Model input shape:",
        model_input.shape
    )

    print(
        "✅ 6-channel input created."
    )


    # --------------------------------------------------------
    # STEP 4: LOAD FINAL CNN MODEL
    # --------------------------------------------------------

    print(
        "\n[4/7] Loading final CNN model..."
    )

    model_path = os.path.join(
        config.BASE_DIR,
        "models",
        "change_detection_cnn_final.keras"
    )

    print(
        "Model path:",
        model_path
    )

    if not os.path.exists(model_path):

        print(
            "\n❌ Final CNN model not found."
        )

        return

    model = load_model(
        model_path,
        compile=False
    )

    print(
        "✅ Final CNN model loaded."
    )


    # --------------------------------------------------------
    # STEP 5: GENERATE PREDICTION
    # --------------------------------------------------------

    print(
        "\n[5/7] Generating change prediction..."
    )

    prediction = model.predict(
        model_input,
        verbose=0
    )[0]

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

    print(
        "Mean prediction:",
        prediction.mean()
    )

    print(
        "✅ Prediction generated."
    )


    # --------------------------------------------------------
    # STEP 6: CREATE BINARY CHANGE MAP
    # --------------------------------------------------------

    print(
        "\n[6/7] Creating change map..."
    )

    threshold = 0.50

    predicted_change = (
        prediction >= threshold
    ).astype(
        np.uint8
    )

    predicted_change = (
        predicted_change.squeeze()
    )

    predicted_pixels = np.sum(
        predicted_change == 1
    )

    print(
        "Threshold:",
        threshold
    )

    print(
        "Predicted change pixels:",
        predicted_pixels
    )

    print(
        "✅ Change map created."
    )


    # --------------------------------------------------------
    # STEP 7: CREATE VISUALIZATION
    # --------------------------------------------------------

    print(
        "\n[7/7] Creating final visualization..."
    )

    # Create overlay

    overlay = processed_b.copy()

    change_mask = (
        predicted_change == 1
    )

    overlay[change_mask] = [
        1.0,
        0.0,
        0.0
    ]

    overlay = (
        0.5 * processed_b
        + 0.5 * overlay
    )


    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_dir = os.path.join(
        config.BASE_DIR,
        "outputs",
        "pipeline"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    plt.figure(
        figsize=(16, 4)
    )


    # Before Image

    plt.subplot(
        1,
        4,
        1
    )

    plt.imshow(
        processed_a
    )

    plt.title(
        "Before Image"
    )

    plt.axis(
        "off"
    )


    # After Image

    plt.subplot(
        1,
        4,
        2
    )

    plt.imshow(
        processed_b
    )

    plt.title(
        "After Image"
    )

    plt.axis(
        "off"
    )


    # Change Map

    plt.subplot(
        1,
        4,
        3
    )

    plt.imshow(
        predicted_change,
        cmap="gray"
    )

    plt.title(
        "Predicted Change"
    )

    plt.axis(
        "off"
    )


    # Overlay

    plt.subplot(
        1,
        4,
        4
    )

    plt.imshow(
        overlay
    )

    plt.title(
        "Change Overlay"
    )

    plt.axis(
        "off"
    )


    plt.tight_layout()


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    output_path = os.path.join(
        output_dir,
        "pipeline_result.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\n✅ Pipeline result saved."
    )

    print(
        "Output path:",
        output_path
    )


    # Display visualization

    plt.show()


    # --------------------------------------------------------
    # PIPELINE COMPLETED
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "END-TO-END PIPELINE COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        "\nInput:",
        filename
    )

    print(
        "Model:",
        "change_detection_cnn_final.keras"
    )

    print(
        "Threshold:",
        threshold
    )

    print(
        "Predicted change pixels:",
        predicted_pixels
    )

    print(
        "\n✅ Satellite image pair successfully processed "
        "from input to final change map."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()