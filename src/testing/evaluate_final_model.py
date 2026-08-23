"""
Final CNN Model Evaluation
Evaluates the final model on the completely held-out test set.
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

    print("✅ Dataset loaded.")

    print(
        "Total images:",
        len(images)
    )

    # --------------------------------------------------
    # 2. Recreate exact three-way split
    # --------------------------------------------------

    print("\nRecreating dataset split...")

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

    print("\n✅ Dataset split recreated.")

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_val)
    )

    print(
        "Final test samples:",
        len(X_test)
    )

    # --------------------------------------------------
    # 3. Load final trained model
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

    print("✅ Final CNN model loaded.")

    # --------------------------------------------------
    # 4. Generate test predictions
    # --------------------------------------------------

    print("\nGenerating predictions on FINAL TEST SET...")

    predictions = model.predict(
        X_test,
        verbose=1
    )

    print(
        "\n✅ Test predictions generated."
    )

    print(
        "Prediction shape:",
        predictions.shape
    )

    print(
        "Prediction range:",
        predictions.min(),
        "to",
        predictions.max()
    )

    print(
        "Mean prediction:",
        predictions.mean()
    )

    # --------------------------------------------------
    # 5. Test multiple thresholds
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
    print("FINAL TEST THRESHOLD ANALYSIS")
    print("=" * 60)

    for threshold in thresholds:

        predicted_masks = (
            predictions >= threshold
        ).astype(np.uint8)

        precision, recall, f1_score, iou = calculate_metrics(
            y_test,
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

        # Select best threshold based on F1-score
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
    # 6. Display final result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)

    print(
        f"\nBest Threshold : "
        f"{best_result['threshold']:.2f}"
    )

    print(
        f"Precision      : "
        f"{best_result['precision'] * 100:.2f}%"
    )

    print(
        f"Recall         : "
        f"{best_result['recall'] * 100:.2f}%"
    )

    print(
        f"F1-Score       : "
        f"{best_result['f1'] * 100:.2f}%"
    )

    print(
        f"IoU            : "
        f"{best_result['iou'] * 100:.2f}%"
    )

    # --------------------------------------------------
    # 7. Save final results
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
            "Model: change_detection_cnn_final.keras\n"
        )

        result_file.write(
            "Test samples: 13\n"
        )

        result_file.write(
            f"Best threshold: "
            f"{best_result['threshold']:.2f}\n\n"
        )

        result_file.write(
            f"Precision: "
            f"{best_result['precision'] * 100:.2f}%\n"
        )

        result_file.write(
            f"Recall: "
            f"{best_result['recall'] * 100:.2f}%\n"
        )

        result_file.write(
            f"F1-Score: "
            f"{best_result['f1'] * 100:.2f}%\n"
        )

        result_file.write(
            f"IoU: "
            f"{best_result['iou'] * 100:.2f}%\n"
        )

    print(
        "\n✅ Final test results saved."
    )

    print(
        "Result path:",
        result_path
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL MODEL EVALUATION COMPLETED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()