"""
Validation Set Detailed Evaluation
"""

import os
import sys
import csv
import numpy as np

# Add src directory to Python path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import config

from training.data_loader import load_dataset
from training.split_dataset import split_dataset
from tensorflow.keras.models import load_model


def calculate_metrics(y_true, y_pred):
    """
    Calculate Precision, Recall, F1-score and IoU.
    """

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    true_positive = np.sum(
        (y_true == 1) & (y_pred == 1)
    )

    false_positive = np.sum(
        (y_true == 0) & (y_pred == 1)
    )

    false_negative = np.sum(
        (y_true == 1) & (y_pred == 0)
    )

    # Precision
    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = true_positive / (
            true_positive + false_positive
        )

    # Recall
    if true_positive + false_negative == 0:
        recall = 0.0
    else:
        recall = true_positive / (
            true_positive + false_negative
        )

    # F1-score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = (
            2 * precision * recall
        ) / (precision + recall)

    # IoU
    union = (
        true_positive
        + false_positive
        + false_negative
    )

    if union == 0:
        iou = 0.0
    else:
        iou = true_positive / union

    return precision, recall, f1_score, iou


def main():

    print("=" * 60)
    print("DETAILED VALIDATION SET EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    images, labels = load_dataset()

    print("✅ Dataset loaded.")

    # --------------------------------------------------
    # 2. Create validation set
    # --------------------------------------------------

    print("\nCreating validation set...")

    X_train, X_val, y_train, y_val = split_dataset(
        images,
        labels,
        test_size=0.2,
        random_state=42
    )

    print("✅ Validation set created.")

    print(
        "Validation images:",
        X_val.shape
    )

    print(
        "Validation labels:",
        y_val.shape
    )

    # --------------------------------------------------
    # 3. Load improved model
    # --------------------------------------------------

    model_path = os.path.join(
        config.BASE_DIR,
        "models",
        "change_detection_cnn_improved.keras"
    )

    print("\nLoading improved CNN...")

    print(
        "Model path:",
        model_path
    )

    if not os.path.exists(model_path):

        print("\n❌ Improved model not found.")

        return

    model = load_model(
        model_path,
        compile=False
    )

    print("✅ Improved CNN loaded.")

    # --------------------------------------------------
    # 4. Generate predictions
    # --------------------------------------------------

    print("\nGenerating predictions for validation set...")

    predictions = model.predict(
        X_val,
        verbose=1
    )

    print("✅ Predictions generated.")

    print(
        "Prediction shape:",
        predictions.shape
    )

    # --------------------------------------------------
    # 5. Convert probabilities to binary masks
    # --------------------------------------------------

    threshold = 0.50

    predicted_masks = (
        predictions >= threshold
    ).astype(np.uint8)

    print(
        "\nUsing prediction threshold:",
        threshold
    )

    # --------------------------------------------------
    # 6. Calculate overall metrics
    # --------------------------------------------------

    overall_precision, overall_recall, overall_f1, overall_iou = (
        calculate_metrics(
            y_val,
            predicted_masks
        )
    )

    # --------------------------------------------------
    # 7. Calculate per-image metrics
    # --------------------------------------------------

    print("\nCalculating per-image metrics...")

    results = []

    for i in range(len(X_val)):

        precision, recall, f1_score, iou = calculate_metrics(
            y_val[i],
            predicted_masks[i]
        )

        changed_pixels = np.sum(
            y_val[i] == 1
        )

        predicted_pixels = np.sum(
            predicted_masks[i] == 1
        )

        results.append({
            "image_number": i + 1,
            "changed_pixels": int(changed_pixels),
            "predicted_change_pixels": int(predicted_pixels),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "iou": iou
        })

    # --------------------------------------------------
    # 8. Calculate average per-image metrics
    # --------------------------------------------------

    average_precision = np.mean([
        result["precision"]
        for result in results
    ])

    average_recall = np.mean([
        result["recall"]
        for result in results
    ])

    average_f1 = np.mean([
        result["f1_score"]
        for result in results
    ])

    average_iou = np.mean([
        result["iou"]
        for result in results
    ])

    # --------------------------------------------------
    # 9. Display per-image results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PER-IMAGE RESULTS")
    print("=" * 60)

    print(
        "\nImage | Precision | Recall | F1-Score | IoU"
    )

    print("-" * 60)

    for result in results:

        print(
            f"{result['image_number']:5d} | "
            f"{result['precision']:.4f}    | "
            f"{result['recall']:.4f} | "
            f"{result['f1_score']:.4f}   | "
            f"{result['iou']:.4f}"
        )

    # --------------------------------------------------
    # 10. Display overall results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("OVERALL VALIDATION RESULTS")
    print("=" * 60)

    print(
        f"\nGlobal Precision : "
        f"{overall_precision * 100:.2f}%"
    )

    print(
        f"Global Recall    : "
        f"{overall_recall * 100:.2f}%"
    )

    print(
        f"Global F1-Score  : "
        f"{overall_f1 * 100:.2f}%"
    )

    print(
        f"Global IoU       : "
        f"{overall_iou * 100:.2f}%"
    )

    print("\nAverage Per-Image Results")

    print(
        f"Average Precision : "
        f"{average_precision * 100:.2f}%"
    )

    print(
        f"Average Recall    : "
        f"{average_recall * 100:.2f}%"
    )

    print(
        f"Average F1-Score  : "
        f"{average_f1 * 100:.2f}%"
    )

    print(
        f"Average IoU       : "
        f"{average_iou * 100:.2f}%"
    )

    # --------------------------------------------------
    # 11. Create outputs directory
    # --------------------------------------------------

    output_dir = os.path.join(
        config.BASE_DIR,
        "outputs"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------
    # 12. Save CSV results
    # --------------------------------------------------

    csv_path = os.path.join(
        output_dir,
        "validation_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as csv_file:

        fieldnames = [
            "image_number",
            "changed_pixels",
            "predicted_change_pixels",
            "precision",
            "recall",
            "f1_score",
            "iou"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    print(
        "\n✅ Per-image results saved."
    )

    print(
        "CSV path:",
        csv_path
    )

    # --------------------------------------------------
    # 13. Save summary text file
    # --------------------------------------------------

    summary_path = os.path.join(
        output_dir,
        "validation_summary.txt"
    )

    with open(
        summary_path,
        "w"
    ) as summary_file:

        summary_file.write(
            "CNN VALIDATION SET EVALUATION\n"
        )

        summary_file.write(
            "=" * 60 + "\n\n"
        )

        summary_file.write(
            "Model: Improved CNN\n"
        )

        summary_file.write(
            "Validation samples: 26\n"
        )

        summary_file.write(
            "Threshold: 0.50\n\n"
        )

        summary_file.write(
            "GLOBAL METRICS\n"
        )

        summary_file.write(
            "-" * 30 + "\n"
        )

        summary_file.write(
            f"Precision: {overall_precision * 100:.2f}%\n"
        )

        summary_file.write(
            f"Recall: {overall_recall * 100:.2f}%\n"
        )

        summary_file.write(
            f"F1-Score: {overall_f1 * 100:.2f}%\n"
        )

        summary_file.write(
            f"IoU: {overall_iou * 100:.2f}%\n\n"
        )

        summary_file.write(
            "AVERAGE PER-IMAGE METRICS\n"
        )

        summary_file.write(
            "-" * 30 + "\n"
        )

        summary_file.write(
            f"Precision: {average_precision * 100:.2f}%\n"
        )

        summary_file.write(
            f"Recall: {average_recall * 100:.2f}%\n"
        )

        summary_file.write(
            f"F1-Score: {average_f1 * 100:.2f}%\n"
        )

        summary_file.write(
            f"IoU: {average_iou * 100:.2f}%\n"
        )

    print(
        "✅ Summary saved."
    )

    print(
        "Summary path:",
        summary_path
    )

    print(
        "\n✅ Detailed validation evaluation completed successfully."
    )


if __name__ == "__main__":
    main()