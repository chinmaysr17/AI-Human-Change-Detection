
import csv
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    BASE_DIR
    / "data"
    / "sentinel2"
    / "analyses"
    / "analysis_acquisition_test_01"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "change_detection_cnn_final.keras"
)

# Save results outside OneDrive.
OUTPUT_DIR = Path(r"C:\Temp\sentinel2_cnn_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (256, 256)

# Experimental thresholds for comparison.
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.50]


# ==================================================
# IMAGE LOADING AND PREPROCESSING
# ==================================================

def load_sentinel_image(image_path):

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    if not np.isfinite(image).all():
        raise ValueError(
            f"Invalid pixel values: {image_path.name}"
        )

    return image


# ==================================================
# SAVE PROBABILITY MAP
# ==================================================

def save_probability_map(prediction):

    probability_image = Image.fromarray(
        np.clip(
            prediction * 255,
            0,
            255
        ).astype(np.uint8)
    )

    output_path = OUTPUT_DIR / "change_probability.png"

    probability_image.save(output_path)

    print("Probability map saved:", output_path)


# ==================================================
# THRESHOLD SENSITIVITY ANALYSIS
# ==================================================

def evaluate_thresholds(prediction, before_image):

    results = []

    total_pixels = prediction.size

    print("\n===================================")
    print(" THRESHOLD SENSITIVITY ANALYSIS")
    print("===================================")

    for threshold in THRESHOLDS:

        # Generate binary change mask.
        change_mask = (
            prediction >= threshold
        ).astype(np.uint8)

        changed_pixels = int(
            np.sum(change_mask)
        )

        change_percentage = (
            changed_pixels / total_pixels
        ) * 100

        # Generate filenames.
        threshold_name = (
            f"{threshold:.2f}".replace(".", "_")
        )

        mask_filename = (
            f"change_mask_{threshold_name}.png"
        )

        overlay_filename = (
            f"change_overlay_{threshold_name}.png"
        )

        # Save binary mask.
        mask_image = Image.fromarray(
            change_mask * 255
        )

        mask_path = OUTPUT_DIR / mask_filename
        mask_image.save(mask_path)

        # Generate red overlay.
        overlay = before_image.copy()

        overlay[change_mask == 1] = (
            0.5 * overlay[change_mask == 1]
            + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)

        overlay_path = OUTPUT_DIR / overlay_filename

        Image.fromarray(overlay).save(overlay_path)

        # Store results.
        results.append({
            "threshold": threshold,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": round(
                change_percentage, 4
            ),
            "mask_file": mask_filename,
            "overlay_file": overlay_filename
        })

        print(
            f"Threshold: {threshold:.2f} | "
            f"Changed pixels: {changed_pixels} | "
            f"Change: {change_percentage:.2f}%"
        )

    return results


# ==================================================
# SAVE CSV REPORT
# ==================================================

def save_csv_report(results):

    csv_path = OUTPUT_DIR / "threshold_comparison.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threshold",
                "changed_pixels",
                "total_pixels",
                "change_percentage",
                "mask_file",
                "overlay_file"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nCSV report saved:", csv_path)


# ==================================================
# MAIN CHANGE DETECTION
# ==================================================

def run_sentinel2_detection():

    print("\n===================================")
    print(" SENTINEL-2 CHANGE DETECTION")
    print("===================================")

    # Validate model.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model not found: {MODEL_PATH}"
        )

    # Input images.
    before_path = IMAGE_DIR / "before.png"
    after_path = IMAGE_DIR / "after_aligned.png"

    print("\nLoading BEFORE image...")
    before = load_sentinel_image(before_path)

    print("Loading ALIGNED AFTER image...")
    after = load_sentinel_image(after_path)

    print("\nImage preprocessing completed.")

    print("Before image shape:", before.shape)
    print("After image shape:", after.shape)

    # Concatenate before and after RGB channels.
    combined = np.concatenate(
        [before, after],
        axis=-1
    )

    model_input = np.expand_dims(
        combined,
        axis=0
    )

    print("Combined model input:", model_input.shape)

    if model_input.shape != (1, 256, 256, 6):
        raise ValueError(
            f"Incorrect CNN input shape: {model_input.shape}"
        )

    # Load trained model.
    print("\nLoading trained CNN model...")

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")

    # Generate prediction only once.
    print("\nGenerating CNN prediction...")

    prediction = model.predict(
        model_input,
        verbose=0
    )[0, :, :, 0]

    if not np.isfinite(prediction).all():
        raise ValueError(
            "CNN prediction contains invalid values."
        )

    print("Prediction completed.")

    # Prediction statistics.
    print("\nPrediction statistics:")
    print("Minimum:", float(prediction.min()))
    print("Maximum:", float(prediction.max()))
    print("Mean:", float(prediction.mean()))

    # Save probability map.
    save_probability_map(prediction)

    # Prepare before image for overlays.
    before_display = np.asarray(
        Image.open(before_path)
        .convert("RGB")
        .resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )
    ).copy()

    # Compare all thresholds.
    results = evaluate_thresholds(
        prediction,
        before_display
    )

    # Save CSV report.
    save_csv_report(results)

    print("\n===================================")
    print(" ANALYSIS COMPLETED")
    print("===================================")

    print("\nGenerated files:")
    print("1. change_probability.png")

    for threshold in THRESHOLDS:
        threshold_name = (
            f"{threshold:.2f}".replace(".", "_")
        )

        print(
            f"2. change_mask_{threshold_name}.png"
        )

        print(
            f"3. change_overlay_{threshold_name}.png"
        )

    print("4. threshold_comparison.csv")

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print(
        "\nNOTE: Threshold comparisons are experimental. "
        "The CNN was trained on LEVIR-CD imagery and "
        "has not been validated for Sentinel-2."
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    run_sentinel2_detection()
