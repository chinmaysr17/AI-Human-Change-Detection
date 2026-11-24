
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


IMAGE_SIZE = (256, 256)


def load_sentinel_pair(before_path, after_path):
    """
    Load and preprocess an aligned Sentinel-2 GeoTIFF pair.

    Returns:
        model_input: (1, 256, 256, 6), float32
        before_rgb: (256, 256, 3), uint8
        after_rgb: (256, 256, 3), uint8
        valid_mask: (256, 256), bool
    """

    before_path = Path(before_path)
    after_path = Path(after_path)

    with rasterio.open(before_path) as before_src, \
         rasterio.open(after_path) as after_src:

        # Verify spatial alignment.
        if before_src.crs != after_src.crs:
            raise ValueError("Before and after CRS do not match.")

        if before_src.width != after_src.width or \
           before_src.height != after_src.height:
            raise ValueError("Before and after dimensions do not match.")

        if before_src.transform != after_src.transform:
            raise ValueError("Before and after spatial transforms do not match.")

        before = before_src.read([1, 2, 3]).astype(np.float32)
        after = after_src.read([1, 2, 3]).astype(np.float32)

        before_valid = np.all(np.isfinite(before), axis=0)
        after_valid = np.all(np.isfinite(after), axis=0)

        if before_src.nodata is not None:
            before_valid &= np.all(
                before != before_src.nodata, axis=0
            )

        if after_src.nodata is not None:
            after_valid &= np.all(
                after != after_src.nodata, axis=0
            )

        valid = before_valid & after_valid

        if not np.any(valid):
            raise ValueError("No common valid pixels in the image pair.")

        # Convert reflectance to display-like RGB values.
        # Use one fixed scale for both dates.
        before_rgb = np.clip(before / 0.4, 0.0, 1.0)
        after_rgb = np.clip(after / 0.4, 0.0, 1.0)

        # Move channels to the final dimension.
        before_rgb = np.moveaxis(before_rgb, 0, -1)
        after_rgb = np.moveaxis(after_rgb, 0, -1)

        # Resize both dates consistently.
        before_resized = np.asarray(
            Image.fromarray(
                (before_rgb * 255).astype(np.uint8)
            ).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        )

        after_resized = np.asarray(
            Image.fromarray(
                (after_rgb * 255).astype(np.uint8)
            ).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        )

        # Resize validity mask with nearest-neighbor interpolation.
        valid_resized = np.asarray(
            Image.fromarray(valid.astype(np.uint8) * 255).resize(
                IMAGE_SIZE,
                Image.Resampling.NEAREST,
            )
        ) > 0

    # Normalize RGB to the model's expected numerical range.
    before_input = before_resized.astype(np.float32) / 255.0
    after_input = after_resized.astype(np.float32) / 255.0

    combined = np.concatenate(
        [before_input, after_input],
        axis=-1,
    )

    model_input = np.expand_dims(combined, axis=0)

    if model_input.shape != (1, 256, 256, 6):
        raise ValueError(
            f"Unexpected model input shape: {model_input.shape}"
        )

    if not np.isfinite(model_input).all():
        raise ValueError("Model input contains invalid values.")

    return (
        model_input,
        before_resized,
        after_resized,
        valid_resized,
    )
