
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    BASE_DIR
    / "data"
    / "sentinel2"
    / "analyses"
    / "analysis_18"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "change_detection_cnn_final.keras"
)

OUTPUT_DIR = Path(r"C:\Temp\boundary_effect_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
PAD = 64
TILE_STEP = 64
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.50]


# ==========================================================
# LOAD SENTINEL IMAGE
# ==========================================================

def load_image(path):

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )

        return np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0


# ==========================================================
# RUN MODEL
# ==========================================================

def predict(model, before, after):

    combined = np.concatenate(
        [before, after],
        axis=-1,
    )

    batch = np.expand_dims(
        combined,
        axis=0,
    )

    return model.predict(
        batch,
        verbose=0,
    )[0, :, :, 0]


# ==========================================================
# REFLECT-PADDED OVERLAPPING PREDICTION
# ==========================================================

def predict_with_padding(model, before, after):

    padded_before = np.pad(
        before,
        ((PAD, PAD), (PAD, PAD), (0, 0)),
        mode="reflect",
    )

    padded_after = np.pad(
        after,
        ((PAD, PAD), (PAD, PAD), (0, 0)),
        mode="reflect",
    )

    padded_size = IMAGE_SIZE + (2 * PAD)

    accumulated = np.zeros(
        (padded_size, padded_size),
        dtype=np.float32,
    )

    counts = np.zeros_like(accumulated)

    for y in range(0, 2 * PAD + 1, TILE_STEP):
        for x in range(0, 2 * PAD + 1, TILE_STEP):

            before_tile = padded_before[
                y:y + IMAGE_SIZE,
                x:x + IMAGE_SIZE,
            ]

            after_tile = padded_after[
                y:y + IMAGE_SIZE,
                x:x + IMAGE_SIZE,
            ]

            prediction = predict(
                model,
                before_tile,
                after_tile,
            )

            # Keep only the central 128x128 region.
            # This excludes each tile's outer 64-pixel border.
            central = prediction[
                PAD:IMAGE_SIZE - PAD,
                PAD:IMAGE_SIZE - PAD,
            ]

            global_y = y + PAD
            global_x = x + PAD

            accumulated[
                global_y:global_y + central.shape[0],
                global_x:global_x + central.shape[1],
            ] += central

            counts[
                global_y:global_y + central.shape[0],
                global_x:global_x + central.shape[1],
            ] += 1

    averaged = np.divide(
        accumulated,
        counts,
        out=np.zeros_like(accumulated),
        where=counts > 0,
    )

    # Extract the original image area.
    return averaged[
        PAD:PAD + IMAGE_SIZE,
        PAD:PAD + IMAGE_SIZE,
    ]


# ==========================================================
# BOUNDARY / INTERIOR STATISTICS
# ==========================================================

def calculate_statistics(prediction, name):

    boundary = np.zeros(
        prediction.shape,
        dtype=bool,
    )

    border = 5

    boundary[:border, :] = True
    boundary[-border:, :] = True
    boundary[:, :border] = True
    boundary[:, -border:] = True

    interior = ~boundary

    print(f"\n{name}")
    print("-" * 50)

    print("Mean probability:", float(prediction.mean()))
    print("Maximum probability:", float(prediction.max()))

    print(
        "Boundary mean:",
        float(prediction[boundary].mean()),
    )

    print(
        "Interior mean:",
        float(prediction[interior].mean()),
    )

    for threshold in THRESHOLDS:

        mask = prediction >= threshold

        boundary_count = int(
            np.sum(mask & boundary)
        )

        interior_count = int(
            np.sum(mask & interior)
        )

        total = boundary_count + interior_count

        boundary_percentage = (
            boundary_count / total * 100
            if total else 0
        )

        print(
            f"Threshold {threshold:.2f}: "
            f"Total={total}, "
            f"Boundary={boundary_count}, "
            f"Interior={interior_count}, "
            f"Boundary%={boundary_percentage:.2f}"
        )


# ==========================================================
# SAVE OUTPUTS
# ==========================================================

def save_outputs(prediction, name):

    probability = Image.fromarray(
        np.clip(
            prediction * 255,
            0,
            255,
        ).astype(np.uint8)
    )

    probability.save(
        OUTPUT_DIR / f"{name}_probability.png"
    )

    for threshold in THRESHOLDS:

        mask = (
            prediction >= threshold
        ).astype(np.uint8) * 255

        Image.fromarray(mask).save(
            OUTPUT_DIR
            / f"{name}_mask_{str(threshold).replace('.', '_')}.png"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    before_path = IMAGE_DIR / "before.png"
    after_path = IMAGE_DIR / "after.png"

    if not before_path.exists() or not after_path.exists():
        raise FileNotFoundError(
            "Sentinel-2 before/after images not found."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    print("Loading Sentinel-2 images...")

    before = load_image(before_path)
    after = load_image(after_path)

    print("Loading Final CNN...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("\nRunning normal prediction...")

    normal_prediction = predict(
        model,
        before,
        after,
    )

    print("Running reflect-padded overlapping prediction...")

    padded_prediction = predict_with_padding(
        model,
        before,
        after,
    )

    difference = np.abs(
        normal_prediction - padded_prediction
    )

    print("\nPREDICTION DIFFERENCE")
    print("-" * 50)
    print("Mean absolute difference:", float(difference.mean()))
    print("Maximum absolute difference:", float(difference.max()))

    calculate_statistics(
        normal_prediction,
        "NORMAL PREDICTION",
    )

    calculate_statistics(
        padded_prediction,
        "REFLECT-PADDED PREDICTION",
    )

    save_outputs(
        normal_prediction,
        "normal",
    )

    save_outputs(
        padded_prediction,
        "reflect_padded",
    )

    Image.fromarray(
        np.clip(
            difference * 255,
            0,
            255,
        ).astype(np.uint8)
    ).save(
        OUTPUT_DIR / "prediction_difference.png"
    )

    print("\nExperiment completed.")
    print("Results saved to:", OUTPUT_DIR)
    print("Original model and application files were not modified.")


if __name__ == "__main__":
    main()
