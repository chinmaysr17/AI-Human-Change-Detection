import numpy as np
import tensorflow as tf
from pathlib import Path

from src.training.mining_data_loader_v2 import (
    load_mining_split,
    calculate_band_statistics,
    standardize_images
)

MODEL_PATH = Path("src/training/models/mining_unet_v2_best.keras")
THRESHOLD = 0.70

BANDS = ["B02", "B03", "B04", "B08"]


def calculate_metrics(y_true, y_pred):

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 0
    )

    return {
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1": f1 * 100,
        "IoU": iou * 100,
        "Coverage": y_pred.mean() * 100
    }


def main():

    print("=" * 65)
    print("V2 SPECTRAL BAND-ABLATION EXPERIMENT")
    print("=" * 65)

    X_train, _ = load_mining_split("train")
    X_val_raw, y_val = load_mining_split("validation")

    mean, std = calculate_band_statistics(X_train)

    X_val = standardize_images(
        X_val_raw, mean, std
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    experiments = [
        ("Baseline - All bands", None),
        ("Without B02", 0),
        ("Without B03", 1),
        ("Without B04", 2),
        ("Without B08", 3)
    ]

    results = []

    for name, removed_band in experiments:

        print("\n" + "-" * 65)
        print(name)
        print("-" * 65)

        X_test = X_val.copy()

        if removed_band is not None:
            X_test[:, :, :, removed_band] = 0.0

        predictions = model.predict(
            X_test,
            batch_size=1,
            verbose=0
        )

        metrics = calculate_metrics(
            y_val,
            predictions >= THRESHOLD
        )

        row = {
            "Experiment": name,
            **metrics
        }

        results.append(row)

        for key, value in metrics.items():
            print(f"{key}: {value:.2f}%")

    print("\n" + "=" * 85)
    print("FINAL BAND-ABLATION COMPARISON")
    print("=" * 85)

    print(
        f"{'Experiment':25s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'F1':>12s}"
        f"{'IoU':>12s}"
    )

    for row in results:

        print(
            f"{row['Experiment']:25s}"
            f"{row['Precision']:11.2f}%"
            f"{row['Recall']:11.2f}%"
            f"{row['F1']:11.2f}%"
            f"{row['IoU']:11.2f}%"
        )

    print("\nExperiment completed.")
    print("Original V2 checkpoint remains unchanged.")


if __name__ == "__main__":
    main()
