import numpy as np
import rasterio
from rasterio.io import MemoryFile
from PIL import Image


IMAGE_SIZE = (256, 256)


def load_sentinel_pair(before_bytes, after_bytes):
    """
    Preprocess an aligned Sentinel-2 GeoTIFF pair directly from memory.

    No temporary image files are created. The MemoryFile datasets are
    closed before this function returns.

    Returns:
        model_input: (1, 256, 256, 6), float32
        before_rgb: (256, 256, 3), uint8
        after_rgb: (256, 256, 3), uint8
        valid_mask: (256, 256), bool
    """
    if not isinstance(before_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("before_bytes must contain GeoTIFF bytes.")

    if not isinstance(after_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("after_bytes must contain GeoTIFF bytes.")

    with MemoryFile(bytes(before_bytes)) as before_mem, \
         MemoryFile(bytes(after_bytes)) as after_mem:
        with before_mem.open() as before_src, after_mem.open() as after_src:
            if before_src.crs != after_src.crs:
                raise ValueError("Before and after CRS do not match.")

            if (before_src.width, before_src.height) != (
                after_src.width, after_src.height
            ):
                raise ValueError("Before and after dimensions do not match.")

            if before_src.transform != after_src.transform:
                raise ValueError("Before and after spatial transforms do not match.")

            if before_src.count < 3 or after_src.count < 3:
                raise ValueError("Both GeoTIFFs must contain at least three RGB bands.")

            before = before_src.read([1, 2, 3], masked=True).astype(np.float32)
            after = after_src.read([1, 2, 3], masked=True).astype(np.float32)

            before_data = np.ma.filled(before, np.nan)
            after_data = np.ma.filled(after, np.nan)
            before_valid = (
                ~np.ma.getmaskarray(before).any(axis=0)
                & np.isfinite(before_data).all(axis=0)
            )
            after_valid = (
                ~np.ma.getmaskarray(after).any(axis=0)
                & np.isfinite(after_data).all(axis=0)
            )
            valid = before_valid & after_valid

            if not np.any(valid):
                raise ValueError("No common valid pixels in the image pair.")

            before_rgb = np.clip(np.nan_to_num(before_data, nan=0.0) / 0.4, 0.0, 1.0)
            after_rgb = np.clip(np.nan_to_num(after_data, nan=0.0) / 0.4, 0.0, 1.0)

            before_rgb = np.moveaxis(before_rgb, 0, -1)
            after_rgb = np.moveaxis(after_rgb, 0, -1)

            before_resized = np.asarray(
                Image.fromarray((before_rgb * 255).astype(np.uint8)).resize(
                    IMAGE_SIZE, Image.Resampling.BILINEAR
                )
            )
            after_resized = np.asarray(
                Image.fromarray((after_rgb * 255).astype(np.uint8)).resize(
                    IMAGE_SIZE, Image.Resampling.BILINEAR
                )
            )
            valid_resized = np.asarray(
                Image.fromarray(valid.astype(np.uint8) * 255).resize(
                    IMAGE_SIZE, Image.Resampling.NEAREST
                )
            ) > 0

    if not np.any(valid_resized):
        raise ValueError("No common valid pixels remain after resizing.")

    before_input = before_resized.astype(np.float32) / 255.0
    after_input = after_resized.astype(np.float32) / 255.0
    combined = np.concatenate([before_input, after_input], axis=-1)
    model_input = np.expand_dims(combined, axis=0)

    if model_input.shape != (1, 256, 256, 6):
        raise ValueError(f"Unexpected model input shape: {model_input.shape}")

    if not np.isfinite(model_input).all():
        raise ValueError("Model input contains invalid values.")

    return model_input, before_resized, after_resized, valid_resized
