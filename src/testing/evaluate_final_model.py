"""
Final CNN Model Evaluation

Evaluation methodology:
1. Recreate the exact 102/13/13 train-validation-test split.
2. Load the final CNN model.
3. Use ONLY the validation set to select the best threshold.
4. Apply the selected threshold ONCE to the held-out test set.
5. Calculate final Precision, Recall, F1-score and IoU.
"""

import os
import sys
import numpy as np

# Add src directory to Python path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import config

from training.data_loader import load_dataset
from training.split_train_val_test import split_train_val_test
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
    print("FINAL CNN MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load complete dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    images, labels = load_dataset()

    print("[OK] Dataset loaded.")
    print("Total images:", len(images))

    # --------------------------------------------------
    # 2. Recreate exact three-way split
    # --------------------------------------------------

    print("\nRecreating exact train/validation/test split...")

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

    print("\n[OK] Dataset split recreated.")

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Final test samples:", len(X_test))

    # --------------------------------------------------
    # 3. Load final trained model
    # --------------------------------------------------

    model_path = os.path.join(
        config.BASE_DIR,
        "models",
        "change_detection_cnn_final.keras"
    )

    print("\nLoading final CNN model...")
    print("Model path:", model_path)

    if not os.path.exists(model_path):

        print("\n[ERROR] Final CNN model not found.")
        print("Expected path:", model_path)
        return

    model = load_model(
        model_path,
        compile=False
    )

    print("[OK] Final CNN model loaded.")

    # --------------------------------------------------
    # 4. Generate validation predictions
    # --------------------------------------------------

    print("\nGenerating predictions on VALIDATION SET...")

    validation_predictions = model.predict(
        X_val,
        verbose=1
    )

    print("\n[OK] Validation predictions generated.")

    print(
        "Validation prediction shape:",
        validation_predictions.shape
    )

    print(
        "Validation prediction range:",
        validation_predictions.min(),
        "to",
        validation_predictions.max()
    )

    print(
        "Validation mean prediction:",
        validation_predictions.mean()
    )

    # --------------------------------------------------
    # 5. Select threshold using VALIDATION SET ONLY
    # --------------------------------------------------

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50
    ]

    best_result = None

    print("\n" + "=" * 60)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 60)

    for threshold in thresholds:

        predicted_masks = (
            validation_predictions >= threshold
        ).astype(np.uint8)

        precision, recall, f1_score, iou = calculate_metrics(
            y_val,
            predicted_masks
        )

        predicted_pixels = np.sum(
            predicted_masks == 1
        )

        print(
            f"\nThreshold: {threshold:.2f}"
        )

        print(
            f"Predicted change pixels: "
            f"{predicted_pixels}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall:    {recall:.4f}"
        )

        print(
            f"F1-Score:  {f1_score:.4f}"
        )

        print(
            f"IoU:       {iou:.4f}"
        )

        # Select best threshold using validation F1
        if (
            best_result is None
            or f1_score > best_result["f1"]
        ):

            best_result = {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "iou": iou
            }

    # --------------------------------------------------
    # 6. Display selected threshold
    # --------------------------------------------------

    selected_threshold = best_result["threshold"]

    print("\n" + "=" * 60)
    print("SELECTED VALIDATION THRESHOLD")
    print("=" * 60)

    print(
        f"\nBest Validation Threshold : "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Validation Precision      : "
        f"{best_result['precision'] * 100:.2f}%"
    )

    print(
        f"Validation Recall         : "
        f"{best_result['recall'] * 100:.2f}%"
    )

    print(
        f"Validation F1-Score       : "
        f"{best_result['f1'] * 100:.2f}%"
    )

    print(
        f"Validation IoU            : "
        f"{best_result['iou'] * 100:.2f}%"
    )

    # --------------------------------------------------
    # 7. Generate FINAL TEST predictions
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)

    print(
        "\nApplying selected threshold to FINAL TEST SET:"
        f" {selected_threshold:.2f}"
    )

    test_predictions = model.predict(
        X_test,
        verbose=1
    )

    print("\n[OK] Final test predictions generated.")

    print(
        "Test prediction shape:",
        test_predictions.shape
    )

    # --------------------------------------------------
    # 8. Apply fixed validation threshold
    # --------------------------------------------------

    test_predicted_masks = (
        test_predictions >= selected_threshold
    ).astype(np.uint8)

    (
        test_precision,
        test_recall,
        test_f1,
        test_iou
    ) = calculate_metrics(
        y_test,
        test_predicted_masks
    )

    test_predicted_pixels = np.sum(
        test_predicted_masks == 1
    )

    test_total_pixels = y_test.size

    print(
        "\nTest predicted change pixels:",
        test_predicted_pixels
    )

    print(
        "Test total pixels:",
        test_total_pixels
    )

    # --------------------------------------------------
    # 9. Display FINAL test results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)

    print(
        f"\nThreshold Used : "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Precision      : "
        f"{test_precision * 100:.2f}%"
    )

    print(
        f"Recall         : "
        f"{test_recall * 100:.2f}%"
    )

    print(
        f"F1-Score       : "
        f"{test_f1 * 100:.2f}%"
    )

    print(
        f"IoU            : "
        f"{test_iou * 100:.2f}%"
    )

    # --------------------------------------------------
    # 10. Save final results
    # --------------------------------------------------

    output_dir = os.path.join(
        config.BASE_DIR,
        "outputs"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    result_path = os.path.join(
        output_dir,
        "final_test_results.txt"
    )

    with open(
        result_path,
        "w"
    ) as result_file:

        result_file.write(
            "FINAL CNN TEST RESULTS\n"
        )

        result_file.write(
            "=" * 60 + "\n\n"
        )

        result_file.write(
            "Evaluation methodology:\n"
        )

        result_file.write(
            "Threshold selected using validation set only.\n"
        )

        result_file.write(
            "Final metrics calculated on held-out test set.\n\n"
        )

        result_file.write(
            "Model: change_detection_cnn_final.keras\n"
        )

        result_file.write(
            "Training samples: 102\n"
        )

        result_file.write(
            "Validation samples: 13\n"
        )

        result_file.write(
            "Test samples: 13\n\n"
        )

        result_file.write(
            f"Selected threshold: "
            f"{selected_threshold:.2f}\n\n"
        )

        result_file.write(
            f"Precision: "
            f"{test_precision * 100:.2f}%\n"
        )

        result_file.write(
            f"Recall: "
            f"{test_recall * 100:.2f}%\n"
        )

        result_file.write(
            f"F1-Score: "
            f"{test_f1 * 100:.2f}%\n"
        )

        result_file.write(
            f"IoU: "
            f"{test_iou * 100:.2f}%\n"
        )

    print(
        "\n[OK] Final test results saved."
    )

    print(
        "Result path:",
        result_path
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()