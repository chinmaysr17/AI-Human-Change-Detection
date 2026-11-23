from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "dataset" / "Mining" / "splits"

IMAGE_SCALE = 10000.0

EXPECTED_IMAGE_SHAPE = (256, 256, 4)
EXPECTED_MASK_SHAPE = (256, 256)

BAND_NAMES = ["B02", "B03", "B04", "B08"]


def load_mining_split(split):
    """Load raw four-band mining images and binary masks."""

    if split not in ("train", "validation", "test"):
        raise ValueError("Invalid dataset split.")

    split_path = SPLIT_DIR / split

    if not split_path.exists():
        raise FileNotFoundError(split_path)

    image_files = sorted(split_path.glob("mining_*.npy"))

    if not image_files:
        raise ValueError(f"No images found in {split_path}")

    images = []
    masks = []
    tile_ids = []

    for image_path in image_files:

        tile_id = image_path.stem.replace("mining_", "")
        mask_path = split_path / f"mask_{tile_id}.npy"

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        image = np.load(image_path)
        mask = np.load(mask_path)

        if image.shape != EXPECTED_IMAGE_SHAPE:
            raise ValueError(f"Invalid image shape: {tile_id}")

        if mask.shape != EXPECTED_MASK_SHAPE:
            raise ValueError(f"Invalid mask shape: {tile_id}")

        if not np.all(np.isfinite(image)):
            raise ValueError(f"Invalid image values: {tile_id}")

        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError(f"Invalid mask values: {tile_id}")

        image = image.astype(np.float32) / IMAGE_SCALE
        mask = mask.astype(np.float32)[..., np.newaxis]

        images.append(image)
        masks.append(mask)
        tile_ids.append(tile_id)

    X = np.stack(images).astype(np.float32)
    Y = np.stack(masks).astype(np.float32)

    print(f"\n{split.upper()} DATASET LOADED")
    print("Samples:", len(X))
    print("Image shape:", X.shape)
    print("Mask shape:", Y.shape)
    print("Tile IDs:", ", ".join(tile_ids))
    print("Positive coverage:", round(float(Y.mean() * 100), 2), "%")

    return X, Y


def calculate_band_statistics(X_train):
    """Calculate per-band statistics using training images only."""

    mean = X_train.mean(axis=(0, 1, 2)).astype(np.float32)
    std = X_train.std(axis=(0, 1, 2)).astype(np.float32)

    std = np.maximum(std, 1e-6)

    print("\nTRAINING-ONLY NORMALIZATION STATISTICS")
    print("=" * 55)

    for i, band in enumerate(BAND_NAMES):
        print(
            f"{band}: mean={mean[i]:.6f}, "
            f"std={std[i]:.6f}"
        )

    return mean, std


def standardize_images(images, mean, std):
    """Apply training-derived per-band standardization."""

    return (
        (images - mean.reshape(1, 1, 1, 4))
        / std.reshape(1, 1, 1, 4)
    ).astype(np.float32)


if __name__ == "__main__":

    X_train, y_train = load_mining_split("train")

    mean, std = calculate_band_statistics(X_train)

    X_train_normalized = standardize_images(
        X_train, mean, std
    )

    print("\nNORMALIZATION CHECK")
    print("Normalized shape:", X_train_normalized.shape)
    print("Normalized minimum:", float(X_train_normalized.min()))
    print("Normalized maximum:", float(X_train_normalized.max()))
    print("All values finite:", bool(np.isfinite(X_train_normalized).all()))

    print("\nV2 DATA LOADER CHECK COMPLETED")
