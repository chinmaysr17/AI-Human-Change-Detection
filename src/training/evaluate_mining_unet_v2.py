from pathlib import Path
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.training.mining_data_loader_v2 import (
    load_mining_split,
    calculate_band_statistics,
    standardize_images
)


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "src" / "training" / "models" / "mining_unet_v2_best.keras"

OUTPUT_DIR = ROOT / "reports" / "mining_v2_validation"

THRESHOLDS = np.round(
    np.arange(0.10, 0.951, 0.05), 2
)


def calculate_metrics(y_true, y_pred):

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.logical_and(y_true, y_pred).sum())
    fp = int(np.logical_and(~y_true, y_pred).sum())
    fn = int(np.logical_and(y_true, ~y_pred).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 0.0
    )

    coverage = float(y_pred.mean() * 100)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "IoU": iou,
        "Dice": dice,
        "predicted_coverage_percent": coverage
    }


def save_csv(path, rows):

    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def create_rgb(image):

    # RGB composite: B04, B03, B02
    rgb = image[:, :, [2, 1, 0]].astype(np.float32)

    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)

    if high <= low:
        return np.zeros_like(rgb)

    rgb = np.clip((rgb - low) / (high - low), 0, 1)

    return rgb


def main():

    print("=" * 65)
    print("MINING U-NET V2 - VALIDATION EVALUATION")
    print("=" * 65)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V2 checkpoint not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training and validation data.
    X_train, _ = load_mining_split("train")
    X_val_raw, y_val = load_mining_split("validation")

    # Recalculate statistics using training data only.
    band_mean, band_std = calculate_band_statistics(X_train)

    # Apply the same normalization used during V2 training.
    X_val = standardize_images(
        X_val_raw,
        band_mean,
        band_std
    )

    # Get validation tile IDs in the loader's sorted order.
    split_dir = ROOT / "dataset" / "Mining" / "splits" / "validation"

    image_files = sorted(split_dir.glob("mining_*.npy"))

    tile_ids = [
        path.stem.replace("mining_", "")
        for path in image_files
    ]

    if len(tile_ids) != len(X_val):
        raise ValueError("Validation tile ID count does not match loaded data.")

    print("\nLoading best V2 checkpoint...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Checkpoint loaded:", MODEL_PATH.name)

    # Predict probability maps.
    print("\nGenerating validation predictions...")

    probabilities = model.predict(
        X_val,
        batch_size=1,
        verbose=1
    )

    if probabilities.shape != y_val.shape:
        raise ValueError(
            f"Prediction shape {probabilities.shape} "
            f"does not match mask shape {y_val.shape}"
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predictions contain non-finite values.")

    print("Prediction shape:", probabilities.shape)
    print(
        "Probability range:",
        float(probabilities.min()),
        "to",
        float(probabilities.max())
    )

    # Evaluate all thresholds using global pixel counts.
    threshold_rows = []

    for threshold in THRESHOLDS:

        predicted_masks = probabilities >= threshold

        metrics = calculate_metrics(
            y_val,
            predicted_masks
        )

        row = {
            "threshold": float(threshold),
            **metrics
        }

        threshold_rows.append(row)

    save_csv(
        OUTPUT_DIR / "threshold_metrics.csv",
        threshold_rows
    )

    # Select the threshold with the highest validation F1.
    # This is a validation-only threshold, not a test-set result.
    best_row = max(
        threshold_rows,
        key=lambda row: row["F1"]
    )

    best_threshold = best_row["threshold"]

    print("\n" + "=" * 65)
    print("GLOBAL VALIDATION RESULTS")
    print("=" * 65)

    print("Selected validation threshold:", best_threshold)

    for key in [
        "precision",
        "recall",
        "F1",
        "IoU",
        "Dice",
        "predicted_coverage_percent"
    ]:
        value = best_row[key]
        if key != "predicted_coverage_percent":
            value *= 100
        print(f"{key}: {value:.2f}%")

    # Per-image metrics at the selected validation threshold.
    per_image_rows = []

    for index, tile_id in enumerate(tile_ids):

        metrics = calculate_metrics(
            y_val[index],
            probabilities[index] >= best_threshold
        )

        per_image_rows.append({
            "tile_id": tile_id,
            **metrics
        })

    save_csv(
        OUTPUT_DIR / "per_tile_metrics.csv",
        per_image_rows
    )

    # Save probability maps without thresholding.
    np.save(
        OUTPUT_DIR / "validation_probability_maps.npy",
        probabilities
    )

    # Save visual comparisons.
    for index, tile_id in enumerate(tile_ids):

        rgb = create_rgb(X_val_raw[index])

        ground_truth = y_val[index, :, :, 0]
        probability = probabilities[index, :, :, 0]
        prediction = probability >= best_threshold

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(rgb)
        axes[0].set_title("Sentinel-2 RGB")

        axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")

        axes[2].imshow(probability, cmap="viridis", vmin=0, vmax=1)
        axes[2].set_title("Predicted Probability")

        axes[3].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title(f"Prediction @ {best_threshold:.2f}")

        for axis in axes:
            axis.axis("off")

        fig.suptitle(f"V2 Validation - Tile {tile_id}")
        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR / f"tile_{tile_id}_comparison.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

    # Save selected threshold and aggregate results.
    with open(
        OUTPUT_DIR / "evaluation_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("MINING U-NET V2 VALIDATION EVALUATION\n")
        file.write("=" * 50 + "\n")
        file.write(f"Checkpoint: {MODEL_PATH.name}\n")
        file.write(f"Validation tiles: {', '.join(tile_ids)}\n")
        file.write(f"Selected threshold: {best_threshold}\n\n")

        for key, value in best_row.items():
            file.write(f"{key}: {value}\n")

    print("\nPER-TILE VALIDATION RESULTS")
    print("=" * 65)

    for row in per_image_rows:
        print(
            f"Tile {row['tile_id']}: "
            f"Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"F1={row['F1']:.4f}, "
            f"IoU={row['IoU']:.4f}"
        )

    print("\nEvaluation files saved in:")
    print(OUTPUT_DIR)

    print("\nV2 VALIDATION EVALUATION COMPLETED")


if __name__ == "__main__":
    main()

