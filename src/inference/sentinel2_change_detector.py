import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from src.preprocessing.sentinel2_preprocessor import load_sentinel_pair

# ==================================================
# PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

# This standalone script reads an existing local pair for offline
# experimentation. The web application retrieves imagery in memory.
IMAGE_DIR = (
    BASE_DIR
    / "data"
    / "sentinel2"
    / "analyses"
    / "analysis_new_york_global_test"
)
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "change_detection_cnn_final.keras"
)

IMAGE_SIZE = (256, 256)

# Experimental thresholds for comparison.
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.50]


# ==================================================
# THRESHOLD SENSITIVITY ANALYSIS
# ==================================================

def evaluate_thresholds(prediction, valid_mask):
    """Calculate threshold metrics without saving masks or reports."""

    results = []
    total_pixels = int(valid_mask.sum())

    if total_pixels == 0:
        raise ValueError("No valid pixels available for evaluation.")

    print("\n===================================")
    print(" THRESHOLD SENSITIVITY ANALYSIS")
    print("===================================")

    for threshold in THRESHOLDS:
        change_mask = (
            (prediction >= threshold) & valid_mask
        ).astype(np.uint8)

        changed_pixels = int(np.sum(change_mask))
        change_percentage = (changed_pixels / total_pixels) * 100.0

        result = {
            "threshold": threshold,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": round(change_percentage, 4),
        }
        results.append(result)

        print(
            f"Threshold: {threshold:.2f} | "
            f"Changed pixels: {changed_pixels} | "
            f"Change: {change_percentage:.2f}%"
        )

    return results


# ==================================================
# MAIN CHANGE DETECTION
# ==================================================

def run_sentinel2_detection():
    print("\n===================================")
    print(" SENTINEL-2 CHANGE DETECTION")
    print("===================================")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"CNN model not found: {MODEL_PATH}")

    before_path = IMAGE_DIR / "before.tif"
    after_path = IMAGE_DIR / "after.tif"

    if not before_path.is_file():
        raise FileNotFoundError(f"Before image not found: {before_path}")
    if not after_path.is_file():
        raise FileNotFoundError(f"After image not found: {after_path}")

    print("\nReading existing GeoTIFF pair into memory...")
    before_bytes = before_path.read_bytes()
    after_bytes = after_path.read_bytes()

    print("Preprocessing GeoTIFF pair in memory...")
    (
        model_input,
        before_display,
        after_display,
        valid_mask,
    ) = load_sentinel_pair(before_bytes, after_bytes)

    # Drop the source byte buffers after preprocessing.
    del before_bytes, after_bytes

    print("\nImage preprocessing completed.")
    print("CNN input shape:", model_input.shape)
    print("Valid pixels:", int(valid_mask.sum()))
    print("Invalid pixels:", int(valid_mask.size - valid_mask.sum()))

    if model_input.shape != (1, 256, 256, 6):
        raise ValueError(
            f"Incorrect CNN input shape: {model_input.shape}"
        )

    print("\nLoading trained CNN model...")
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")

    print("\nGenerating CNN prediction...")
    prediction = model.predict(model_input, verbose=0)[0, :, :, 0]

    if not np.isfinite(prediction).all():
        raise ValueError("CNN prediction contains invalid values.")

    print("Prediction completed.")
    print("\nPrediction statistics:")
    print("Minimum:", float(prediction.min()))
    print("Maximum:", float(prediction.max()))
    print("Mean:", float(prediction.mean()))

    results = evaluate_thresholds(prediction, valid_mask)

    print("\n===================================")
    print(" ANALYSIS COMPLETED")
    print("===================================")
    print("No image or report files were written by this script.")
    print(
        "\nNOTE: Threshold comparisons are experimental. "
        "The CNN was trained on LEVIR-CD imagery and "
        "has not been validated for Sentinel-2."
    )

    return results


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    run_sentinel2_detection()
