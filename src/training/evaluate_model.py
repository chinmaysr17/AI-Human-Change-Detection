"""
Compare Baseline and Improved CNN Models
"""

import os
import sys
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

    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = true_positive / (
            true_positive + false_positive
        )

    if true_positive + false_negative == 0:
        recall = 0.0
    else:
        recall = true_positive / (
            true_positive + false_negative
        )

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = (
            2 * precision * recall
        ) / (precision + recall)

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


def evaluate_model(model_path, model_name, X_val, y_val):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    if not os.path.exists(model_path):

        print("\n❌ Model not found:")
        print(model_path)

        return None

    print("\nLoading model...")
    print(model_path)

    # compile=False is important for the improved model
    # because it uses a custom training loss function.
    model = load_model(
        model_path,
        compile=False
    )

    print("✅ Model loaded.")

    # --------------------------------------------------
    # Generate predictions
    # --------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(
        X_val,
        verbose=1
    )

    print("✅ Predictions generated.")

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
    # Test different thresholds
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

    print("\nThreshold Results")
    print("-" * 60)

    for threshold in thresholds:

        predicted_masks = (
            predictions >= threshold
        ).astype(np.uint8)

        precision, recall, f1_score, iou = calculate_metrics(
            y_val,
            predicted_masks
        )

        predicted_pixels = np.sum(
            predicted_masks == 1
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"Predicted pixels: {predicted_pixels:6d} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1_score:.4f} | "
            f"IoU: {iou:.4f}"
        )

        # Select the threshold with the highest F1-score
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

    return best_result


def main():

    print("=" * 60)
    print("CNN MODEL COMPARISON")
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
    # 3. Define model paths
    # --------------------------------------------------

    baseline_path = os.path.join(
        config.BASE_DIR,
        "models",
        "change_detection_cnn.keras"
    )

    improved_path = os.path.join(
        config.BASE_DIR,
        "models",
        "change_detection_cnn_improved.keras"
    )

    # --------------------------------------------------
    # 4. Evaluate baseline model
    # --------------------------------------------------

    baseline_result = evaluate_model(
        baseline_path,
        "BASELINE CNN",
        X_val,
        y_val
    )

    # --------------------------------------------------
    # 5. Evaluate improved model
    # --------------------------------------------------

    improved_result = evaluate_model(
        improved_path,
        "IMPROVED CNN",
        X_val,
        y_val
    )

    # --------------------------------------------------
    # 6. Final comparison
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    if baseline_result is not None:

        print("\nBASELINE CNN")
        print("-" * 30)

        print(
            f"Best Threshold : "
            f"{baseline_result['threshold']:.2f}"
        )

        print(
            f"Precision      : "
            f"{baseline_result['precision'] * 100:.2f}%"
        )

        print(
            f"Recall         : "
            f"{baseline_result['recall'] * 100:.2f}%"
        )

        print(
            f"F1-Score       : "
            f"{baseline_result['f1'] * 100:.2f}%"
        )

        print(
            f"IoU            : "
            f"{baseline_result['iou'] * 100:.2f}%"
        )

    if improved_result is not None:

        print("\nIMPROVED CNN")
        print("-" * 30)

        print(
            f"Best Threshold : "
            f"{improved_result['threshold']:.2f}"
        )

        print(
            f"Precision      : "
            f"{improved_result['precision'] * 100:.2f}%"
        )

        print(
            f"Recall         : "
            f"{improved_result['recall'] * 100:.2f}%"
        )

        print(
            f"F1-Score       : "
            f"{improved_result['f1'] * 100:.2f}%"
        )

        print(
            f"IoU            : "
            f"{improved_result['iou'] * 100:.2f}%"
        )

    # --------------------------------------------------
    # 7. Determine best model
    # --------------------------------------------------

    if (
        baseline_result is not None
        and improved_result is not None
    ):

        print("\n" + "=" * 60)
        print("BEST MODEL")
        print("=" * 60)

        if (
            improved_result["f1"]
            > baseline_result["f1"]
        ):

            print(
                "\n🏆 Improved CNN has the better F1-score."
            )

        elif (
            baseline_result["f1"]
            > improved_result["f1"]
        ):

            print(
                "\n🏆 Baseline CNN has the better F1-score."
            )

        else:

            print(
                "\n🤝 Both models have the same F1-score."
            )

    print(
        "\n✅ Model comparison completed successfully."
    )


if __name__ == "__main__":
    main()
    