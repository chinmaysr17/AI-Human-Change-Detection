from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "dataset" / "Mining" / "splits"
MODEL_PATH = ROOT / "src" / "training" / "models" / "mining_unet_v3_best.keras"

# Values calculated from the V3 training split.
BAND_MEAN = np.array(
    [0.044459, 0.067258, 0.074175, 0.226256],
    dtype=np.float32
)

BAND_STD = np.array(
    [0.032478, 0.036938, 0.059535, 0.072096],
    dtype=np.float32
)

IMAGE_SCALE = 10000.0
PREDICTION_THRESHOLD = 0.70
VALID_SPLITS = {"train", "validation", "test"}


@lru_cache(maxsize=1)
def load_mining_model():
    """Load and cache the trained V3 mining segmentation model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Mining model checkpoint not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


def get_available_mining_tiles(split):
    """Return available tile IDs for a valid dataset split."""
    if split not in VALID_SPLITS:
        raise ValueError(
            "Invalid split. Choose train, validation, or test."
        )

    split_path = SPLIT_DIR / split

    if not split_path.is_dir():
        raise FileNotFoundError(
            f"Mining split directory not found: {split_path}"
        )

    tile_ids = []

    for image_path in split_path.glob("mining_*.npy"):
        match = re.fullmatch(r"mining_(\d+)", image_path.stem)

        if match:
            tile_ids.append(match.group(1))

    return sorted(tile_ids, key=int)


def create_rgb_preview(image):
    """
    Create an RGB preview using B04, B03, B02.
    Per-tile stretching is only for visualization, not model input.
    """
    rgb = image[:, :, [2, 1, 0]].astype(np.float32)

    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)

    if high <= low:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    rgb = (rgb - low) / (high - low)
    rgb = np.clip(rgb, 0.0, 1.0)

    return (rgb * 255).astype(np.uint8)


def run_mining_detection(split, tile_id):
    """Run V3 mining-area segmentation on one existing dataset chip."""
    if split not in VALID_SPLITS:
        raise ValueError(
            "Invalid split. Choose train, validation, or test."
        )

    tile_id = str(tile_id)

    if not re.fullmatch(r"\d+", tile_id):
        raise ValueError("Tile ID must contain digits only.")

    split_path = SPLIT_DIR / split
    image_path = split_path / f"mining_{tile_id}.npy"

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Tile {tile_id} was not found in the {split} split."
        )

    image = np.load(image_path, allow_pickle=False)

    if image.shape != (256, 256, 4):
        raise ValueError(
            f"Expected image shape (256, 256, 4), got {image.shape}."
        )

    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("Image contains non-numeric data.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values.")

    # Match the V3 training preprocessing exactly.
    image = image.astype(np.float32) / IMAGE_SCALE
    standardized = (
        (image - BAND_MEAN.reshape(1, 1, 4))
        / BAND_STD.reshape(1, 1, 4)
    ).astype(np.float32)

    model = load_mining_model()

    probabilities = model.predict(
        np.expand_dims(standardized, axis=0),
        verbose=0
    )[0, :, :, 0]

    binary_mask = (
        probabilities >= PREDICTION_THRESHOLD
    ).astype(np.uint8)

    predicted_pixels = int(binary_mask.sum())
    total_pixels = int(binary_mask.size)
    coverage_percentage = (
        predicted_pixels / total_pixels
    ) * 100.0

    return {
        "tile_id": tile_id,
        "split": split,
        "threshold": PREDICTION_THRESHOLD,
        "predicted_pixels": predicted_pixels,
        "total_pixels": total_pixels,
        "coverage_percentage": round(coverage_percentage, 2),
        "rgb_image": create_rgb_preview(image),
        "binary_mask": binary_mask * 255,
        "probability_map": (
            np.clip(probabilities, 0.0, 1.0) * 255
        ).astype(np.uint8)
    }
