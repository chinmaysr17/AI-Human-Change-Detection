
"""
Fair comparison of existing CNN models.
All models are evaluated on the same untouched LEVIR-CD test split.
Uses the fixed threshold of 0.50.
"""

import os
import sys
import numpy as np
import tensorflow as tf

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from training.data_loader import load_dataset
from training.split_train_val_test import split_train_val_test


def calculate_metrics(y_true, y_pred):
    y_true = y_true.astype(bool).ravel()
    y_pred = y_pred.astype(bool).ravel()

    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    union = tp + fp + fn
    iou = tp / union if union else 0.0

    return precision, recall, f1, iou


def main():
    print("=" * 65)
    print("FAIR CNN MODEL COMPARISON - HELD-OUT TEST SET")
    print("=" * 65)

    print("\nLoading LEVIR-CD dataset...")
    images, labels = load_dataset()

    print("\nRecreating original 80/10/10 split...")

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

    print("\nReserved test samples:", len(X_test))
    print("Test input shape:", X_test.shape)
    print("Test label shape:", y_test.shape)

    threshold = 0.50

    model_files = [
        "change_detection_cnn.keras",
        "change_detection_cnn_final.keras",
        "change_detection_cnn_improved.keras"
    ]

    results = []

    for filename in model_files:
        model_path = os.path.join(
            os.path.dirname(SRC_DIR),
            "models",
            filename
        )

        print("\n" + "=" * 65)
        print("Evaluating:", filename)
        print("=" * 65)

        if not os.path.exists(model_path):
            print("Model file not found. Skipping.")
            continue

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        if model.input_shape[1:] != X_test.shape[1:]:
            raise ValueError(
                f"Input shape mismatch for {filename}: "
                f"{model.input_shape}"
            )

        predictions = model.predict(
            X_test,
            verbose=1
        )

        predicted_masks = (
            predictions >= threshold
        ).astype(np.uint8)

        precision, recall, f1, iou = calculate_metrics(
            y_test,
            predicted_masks
        )

        result = {
            "model": filename,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou
        }

        results.append(result)

        print("\nTest metrics at threshold 0.50")
        print(f"Precision : {precision * 100:.2f}%")
        print(f"Recall    : {recall * 100:.2f}%")
        print(f"F1-score  : {f1 * 100:.2f}%")
        print(f"IoU       : {iou * 100:.2f}%")

        del model
        del predictions
        del predicted_masks

        tf.keras.backend.clear_session()

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    for result in results:
        print("\n", result["model"])
        print(f"Precision: {result['precision'] * 100:.2f}%")
        print(f"Recall   : {result['recall'] * 100:.2f}%")
        print(f"F1       : {result['f1'] * 100:.2f}%")
        print(f"IoU      : {result['iou'] * 100:.2f}%")

    if results:
        best = max(results, key=lambda item: item["f1"])

        print("\nBest F1 on this held-out LEVIR-CD test set:")
        print(best["model"])
        print(f"F1-score: {best['f1'] * 100:.2f}%")

    print("\nComparison completed.")
    print("No model files were modified.")


if __name__ == "__main__":
    main()
