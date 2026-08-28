"""
End-to-End Human-Induced Change Detection Pipeline

Flow:
    Image Pair
        ↓
    Image Loading
        ↓
    Input Validation
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
        ↓
    Quantitative Result Report
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. ADD SRC DIRECTORY TO PYTHON PATH
# ============================================================

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SRC_DIR not in sys.path:
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

    # ========================================================
    # STEP 1: GET IMAGE FILENAME
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "AI-Based Human-Induced "
            "Change Detection"
        )
    )

    parser.add_argument(
        "filename",
        help=(
            "Name of the satellite image pair "
            "e.g. test_1.png"
        )
    )

    args = parser.parse_args()

    filename = args.filename.strip()

    print(
        "\n[1/8] Loading satellite image pair..."
    )

    print(
        "Image pair:",
        filename
    )

    # ========================================================
    # BUILD DATASET PATHS
    # ========================================================

    dataset_dir = os.path.join(
        config.BASE_DIR,
        "dataset",
        "LEVIR-CD"
    )

    image_a_path = os.path.join(
        dataset_dir,
        "A",
        filename
    )

    image_b_path = os.path.join(
        dataset_dir,
        "B",
        filename
    )

    label_path = os.path.join(
        dataset_dir,
        "label",
        filename
    )

    print(
        "\nChecking input files..."
    )

    # ========================================================
    # CHECK IMAGE A
    # ========================================================

    if not os.path.exists(image_a_path):

        print(
            "\n[ERROR] Image A not found:"
        )

        print(
            image_a_path
        )

        return 1

    print(
        "[OK] Image A found."
    )

    # ========================================================
    # CHECK IMAGE B
    # ========================================================

    if not os.path.exists(image_b_path):

        print(
            "\n[ERROR] Image B not found:"
        )

        print(
            image_b_path
        )

        return 1

    print(
        "[OK] Image B found."
    )

    # ========================================================
    # CHECK LABEL
    # ========================================================

    if not os.path.exists(label_path):

        print(
            "\n[ERROR] Ground truth label not found:"
        )

        print(
            label_path
        )

        return 1

    print(
        "[OK] Ground truth label found."
    )

    # ========================================================
    # LOAD IMAGE PAIR
    # ========================================================

    image_a, image_b, label = load_image_pair(
        filename
    )

    if (
        image_a is None
        or image_b is None
        or label is None
    ):

        print(
            "\n[ERROR] Image pair could not be loaded."
        )

        return 1

    print(
        "[OK] Image A loaded."
    )

    print(
        "[OK] Image B loaded."
    )

    print(
        "[OK] Ground truth label loaded."
    )

    # ========================================================
    # CHECK ORIGINAL IMAGE DIMENSIONS
    # ========================================================

    if image_a.shape[:2] != image_b.shape[:2]:

        print(
            "\n[ERROR] Image A and Image B "
            "dimensions do not match."
        )

        print(
            "Image A:",
            image_a.shape
        )

        print(
            "Image B:",
            image_b.shape
        )

        return 1

    if image_a.shape[:2] != label.shape[:2]:

        print(
            "\n[ERROR] Image and label "
            "dimensions do not match."
        )

        print(
            "Image:",
            image_a.shape
        )

        print(
            "Label:",
            label.shape
        )

        return 1

    print(
        "[OK] Original image dimensions verified."
    )

    print(
        "Original dimensions:",
        image_a.shape[:2]
    )

    # ========================================================
    # STEP 2: PREPROCESS IMAGES
    # ========================================================

    print(
        "\n[2/8] Preprocessing images..."
    )

    processed_a, processed_b, processed_label = (
        preprocess_images(
            image_a,
            image_b,
            label
        )
    )

    print(
        "[OK] Preprocessing completed."
    )

    print(
        "Image A shape:",
        processed_a.shape
    )

    print(
        "Image B shape:",
        processed_b.shape
    )

    if processed_label is not None:

        print(
            "Label shape:",
            processed_label.shape
        )

    # ========================================================
    # STEP 3: CREATE 6-CHANNEL INPUT
    # ========================================================

    print(
        "\n[3/8] Creating 6-channel model input..."
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

    # ========================================================
    # VERIFY SIX CHANNELS
    # ========================================================

    if combined_image.shape[-1] != 6:

        print(
            "\n[ERROR] Expected 6-channel input."
        )

        print(
            "Actual channels:",
            combined_image.shape[-1]
        )

        return 1

    # ========================================================
    # ADD BATCH DIMENSION
    # ========================================================

    model_input = np.expand_dims(
        combined_image,
        axis=0
    )

    print(
        "Model input shape:",
        model_input.shape
    )

    print(
        "[OK] 6-channel input created."
    )

    # ========================================================
    # STEP 4: LOAD FINAL CNN MODEL
    # ========================================================

    print(
        "\n[4/8] Loading final CNN model..."
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
            "\n[ERROR] Final CNN model not found."
        )

        print(
            model_path
        )

        return 1

    model = load_model(
        model_path,
        compile=False
    )

    print(
        "[OK] Final CNN model loaded."
    )

    # ========================================================
    # STEP 5: GENERATE PREDICTION
    # ========================================================

    print(
        "\n[5/8] Generating change prediction..."
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
        "[OK] Prediction generated."
    )

    # ========================================================
    # STEP 6: CREATE BINARY CHANGE MAP
    # ========================================================

    print(
        "\n[6/8] Creating binary change map..."
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

    # ========================================================
    # VERIFY OUTPUT SIZE
    # ========================================================

    if predicted_change.shape != processed_b.shape[:2]:

        print(
            "\n[ERROR] Prediction dimensions "
            "do not match processed image."
        )

        print(
            "Prediction:",
            predicted_change.shape
        )

        print(
            "Image:",
            processed_b.shape[:2]
        )

        return 1

    # ========================================================
    # CALCULATE CHANGE STATISTICS
    # ========================================================

    predicted_pixels = int(
        np.sum(
            predicted_change == 1
        )
    )

    total_pixels = int(
        predicted_change.size
    )

    change_percentage = (
        predicted_pixels
        / total_pixels
    ) * 100

    print(
        "Threshold:",
        threshold
    )

    print(
        "Predicted change pixels:",
        predicted_pixels
    )

    print(
        "Total pixels:",
        total_pixels
    )

    print(
        f"Predicted change percentage: "
        f"{change_percentage:.2f}%"
    )

    print(
        "[OK] Change map created."
    )

    # ========================================================
    # STEP 7: CREATE CHANGE OVERLAY
    # ========================================================

    print(
        "\n[7/8] Creating final visualization..."
    )

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

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    output_dir = os.path.join(
        config.BASE_DIR,
        "outputs",
        "pipeline"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ========================================================
    # CREATE UNIQUE FILENAMES
    # ========================================================

    base_name = os.path.splitext(
        filename
    )[0]

    output_path = os.path.join(
        output_dir,
        f"{base_name}_pipeline_result.png"
    )

    report_path = os.path.join(
        output_dir,
        f"{base_name}_pipeline_report.txt"
    )

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    plt.figure(
        figsize=(16, 4)
    )

    # --------------------------------------------------------
    # BEFORE IMAGE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # AFTER IMAGE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PREDICTED CHANGE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CHANGE OVERLAY
    # --------------------------------------------------------

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

    # ========================================================
    # SAVE VISUALIZATION
    # ========================================================

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        "\n[OK] Pipeline visualization saved."
    )

    print(
        "Output path:",
        output_path
    )

    # ========================================================
    # STEP 8: SAVE QUANTITATIVE RESULT REPORT
    # ========================================================

    print(
        "\n[8/8] Saving quantitative result report..."
    )

    report_lines = [
        "AI-BASED HUMAN-INDUCED CHANGE DETECTION",
        "PIPELINE RESULT REPORT",
        "=" * 60,
        "",
        f"Image Pair           : {filename}",
        "Model                : change_detection_cnn_final.keras",
        f"Threshold            : {threshold:.2f}",
        "",
        f"Original Image Size  : "
        f"{image_a.shape[1]} x {image_a.shape[0]}",
        f"Processed Image Size : "
        f"{processed_a.shape[1]} x {processed_a.shape[0]}",
        "",
        f"Total Pixels         : {total_pixels}",
        f"Predicted Change     : {predicted_pixels}",
        f"Change Area          : "
        f"{change_percentage:.2f}%",
        "",
        f"Prediction Minimum   : "
        f"{prediction.min():.6f}",
        f"Prediction Maximum   : "
        f"{prediction.max():.6f}",
        f"Prediction Mean      : "
        f"{prediction.mean():.6f}",
        "",
        f"Visualization        : {output_path}",
        f"Report               : {report_path}",
        "",
        "=" * 60,
        "PIPELINE COMPLETED SUCCESSFULLY",
    ]

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report_file:

        report_file.write(
            "\n".join(
                report_lines
            )
        )

    print(
        "[OK] Quantitative result report saved."
    )

    print(
        "Report path:",
        report_path
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

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
        "\nInput image pair:",
        filename
    )

    print(
        "Model:",
        "change_detection_cnn_final.keras"
    )

    print(
        f"Threshold: {threshold:.2f}"
    )

    print(
        "Predicted change pixels:",
        predicted_pixels
    )

    print(
        f"Predicted change percentage: "
        f"{change_percentage:.2f}%"
    )

    print(
        "\nVisualization:",
        output_path
    )

    print(
        "Result report:",
        report_path
    )

    print(
        "\n[OK] Satellite image pair successfully "
        "processed from input to final change map."
    )

    return 0


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    exit_code = main()

    sys.exit(
        exit_code
    )