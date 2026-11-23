from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "dataset" / "Mining" / "splits"

IMAGE_SCALE = 10000.0
EXPECTED_IMAGE_SHAPE = (256, 256, 4)
EXPECTED_MASK_SHAPE = (256, 256)


def load_mining_split(split):
    """
    Load one mining dataset split.

    Args:
        split: train, validation, or test

    Returns:
        images: float32 array (N, 256, 256, 4)
        masks: float32 array (N, 256, 256, 1)
    """

    if split not in ("train", "validation", "test"):
        raise ValueError(
            "Split must be train, validation, or test."
        )

    split_path = SPLIT_DIR / split

    if not split_path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {split_path}"
        )

    image_files = sorted(split_path.glob("mining_*.npy"))

    if not image_files:
        raise ValueError(
            f"No mining images found in {split_path}"
        )

    images = []
    masks = []

    for image_path in image_files:

        tile_id = image_path.stem.replace("mining_", "")
        mask_path = split_path / f"mask_{tile_id}.npy"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for tile {tile_id}"
            )

        image = np.load(image_path)
        mask = np.load(mask_path)

        if image.shape != EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"Invalid image shape for {tile_id}: {image.shape}"
            )

        if mask.shape != EXPECTED_MASK_SHAPE:
            raise ValueError(
                f"Invalid mask shape for {tile_id}: {mask.shape}"
            )

        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError(
                f"Mask contains unexpected values for tile {tile_id}"
            )

        if not np.all(np.isfinite(image)):
            raise ValueError(
                f"Image contains invalid values for tile {tile_id}"
            )

        # Convert Sentinel-2 reflectance to float32.
        # Preserve values above 1; do not clip.
        image = image.astype(np.float32) / IMAGE_SCALE

        mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        images.append(image)
        masks.append(mask)

    X = np.stack(images).astype(np.float32)
    Y = np.stack(masks).astype(np.float32)

    print(f"\n{split.upper()} DATASET LOADED")
    print("-" * 45)
    print("Samples:", len(X))
    print("Image shape:", X.shape)
    print("Mask shape:", Y.shape)
    print("Image data type:", X.dtype)
    print("Mask data type:", Y.dtype)
    print("Image minimum:", float(X.min()))
    print("Image maximum:", float(X.max()))
    print("Positive mask coverage:", float(Y.mean() * 100), "%")

    return X, Y


if __name__ == "__main__":

    for split_name in ("train", "validation", "test"):
        load_mining_split(split_name)
