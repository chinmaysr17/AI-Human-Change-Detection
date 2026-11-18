
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model


# Project directories
BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = BASE_DIR / "data" / "sentinel2" / "test_shivamogga"
MODEL_PATH = BASE_DIR / "models" / "change_detection_cnn_final.keras"

OUTPUT_DIR = IMAGE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.50


def load_sentinel_image(image_path):
    """Load an RGB Sentinel-2 image and prepare it for the CNN."""

    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image = np.asarray(image, dtype=np.float32) / 255.0

    return image


def run_sentinel2_detection():

    print("Loading Sentinel-2 images...")

    before = load_sentinel_image(
        IMAGE_DIR / "before.png"
    )

    after = load_sentinel_image(
        IMAGE_DIR / "after.png"
    )

    print("Before image:", before.shape)
    print("After image:", after.shape)

    # Combine before and after images
    combined = np.concatenate(
        [before, after],
        axis=-1
    )

    # Add batch dimension
    model_input = np.expand_dims(
        combined,
        axis=0
    )

    print("Model input:", model_input.shape)

    # Load trained CNN
    print("Loading trained CNN...")

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    # Generate prediction
    print("Predicting changes...")

    prediction = model.predict(
        model_input,
        verbose=0
    )[0, :, :, 0]

    # Generate binary change mask
    change_mask = (
        prediction >= THRESHOLD
    ).astype(np.uint8)

    # Save probability map
    probability_image = Image.fromarray(
        np.clip(prediction * 255, 0, 255).astype(np.uint8)
    )

    probability_image.save(
        OUTPUT_DIR / "change_probability.png"
    )

    # Save binary mask
    mask_image = Image.fromarray(
        change_mask * 255
    )

    mask_image.save(
        OUTPUT_DIR / "change_mask.png"
    )

    changed_pixels = int(np.sum(change_mask))
    total_pixels = int(change_mask.size)

    change_percentage = (
        changed_pixels / total_pixels
    ) * 100

    print("\n--- Sentinel-2 Change Detection ---")
    print("Changed pixels:", changed_pixels)
    print("Total pixels:", total_pixels)
    print(f"Predicted change percentage: {change_percentage:.2f}%")
    print("Prediction minimum:", float(prediction.min()))
    print("Prediction maximum:", float(prediction.max()))
    print("Prediction mean:", float(prediction.mean()))

    print("\nResults saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    run_sentinel2_detection()
