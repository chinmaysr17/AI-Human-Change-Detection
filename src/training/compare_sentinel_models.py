
from pathlib import Path
import gc

import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = BASE_DIR / "data" / "sentinel2" / "analyses" / "analysis_18"

MODEL_FILES = [
    "change_detection_cnn_final.keras",
    "change_detection_cnn_improved.keras",
]

OUTPUT_DIR = Path(r"C:\Temp\sentinel2_model_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.50
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.50]


def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR,
        )
        return np.asarray(image, dtype=np.float32) / 255.0


def main():
    before_path = IMAGE_DIR / "before.png"
    after_path = IMAGE_DIR / "after.png"

    if not before_path.exists() or not after_path.exists():
        raise FileNotFoundError("Sentinel-2 image pair not found.")

    before = load_image(before_path)
    after = load_image(after_path)

    model_input = np.expand_dims(
        np.concatenate([before, after], axis=-1),
        axis=0,
    )

    print("Sentinel-2 model comparison")
    print("Input shape:", model_input.shape)
    print("Threshold:", THRESHOLD)

    predictions = {}

    for filename in MODEL_FILES:
        model_path = BASE_DIR / "models" / filename

        print("\nLoading:", filename)

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

        prediction = model.predict(
            model_input,
            verbose=0,
        )[0, :, :, 0]

        predictions[filename] = prediction.copy()

        print("Prediction shape:", prediction.shape)
        print("Minimum:", float(prediction.min()))
        print("Maximum:", float(prediction.max()))
        print("Mean:", float(prediction.mean()))
        print("Standard deviation:", float(prediction.std()))

        for threshold in THRESHOLDS:
            mask = prediction >= threshold
            count = int(mask.sum())
            percentage = count / mask.size * 100

            print(
                f"Threshold {threshold:.2f}: "
                f"{count} pixels ({percentage:.4f}%)"
            )

        mask = prediction >= THRESHOLD
        mask_name = filename.replace(".keras", "")

        Image.fromarray(
            np.clip(prediction * 255, 0, 255).astype(np.uint8)
        ).save(OUTPUT_DIR / f"{mask_name}_probability.png")

        Image.fromarray(
            mask.astype(np.uint8) * 255
        ).save(OUTPUT_DIR / f"{mask_name}_mask.png")

        overlay = (before * 255).astype(np.uint8).copy()
        overlay[mask] = (
            0.5 * overlay[mask]
            + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)

        Image.fromarray(overlay).save(
            OUTPUT_DIR / f"{mask_name}_overlay.png"
        )

        del model
        gc.collect()

    print("\nMODEL-TO-MODEL COMPARISON")
    first, second = MODEL_FILES

    difference = np.abs(
        predictions[first] - predictions[second]
    )

    print("Mean absolute prediction difference:", float(difference.mean()))
    print("Maximum prediction difference:", float(difference.max()))

    print("\nSaved comparison images to:", OUTPUT_DIR)
    print("No original model files or application files were modified.")


if __name__ == "__main__":
    main()
