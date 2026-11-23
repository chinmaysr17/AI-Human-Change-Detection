"""
Create exploratory candidate-change masks from the January 2016–2026
Sentinel-2 mean absolute spectral difference raster.

This is a screening product, NOT a validated human-induced change map.
No source rasters are modified.
"""

from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "sentinel2"

DIFF_PATH = (
    DATA_DIR / "analyses" / "shivamogga_20160113_20260120_spectral"
    / "sentinel2_mean_absolute_difference.tif"
)
AFTER_PATH = DATA_DIR / "aligned" / "after_aligned.tif"

OUTPUT_DIR = (
    DATA_DIR / "analyses" / "shivamogga_20160113_20260120_candidates"
)

NODATA_FALLBACK = -9999.0
PERCENTILES = [95, 98]


def stretch_rgb(bands, valid):
    """Convert B04/B03/B02 reflectance bands to display RGB."""
    # Input order is B04, B03, B02; display order is B02, B03, B04.
    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)
    display = np.zeros(rgb.shape, dtype=np.float32)

    for channel in range(3):
        values = rgb[:, :, channel][valid]
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            high = low + 1e-6
        display[:, :, channel] = np.clip(
            (rgb[:, :, channel] - low) / (high - low), 0, 1
        )

    return (display * 255).astype(np.uint8)


def main():
    if not DIFF_PATH.exists():
        raise FileNotFoundError(f"Difference raster not found: {DIFF_PATH}")
    if not AFTER_PATH.exists():
        raise FileNotFoundError(f"AFTER raster not found: {AFTER_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(DIFF_PATH) as src:
        diff = src.read(1).astype(np.float32)
        diff_profile = src.profile.copy()
        diff_nodata = src.nodata if src.nodata is not None else NODATA_FALLBACK

    with rasterio.open(AFTER_PATH) as src:
        after = src.read().astype(np.float32)
        after_profile = src.profile.copy()
        after_nodata = src.nodata if src.nodata is not None else NODATA_FALLBACK

    if diff.shape != after.shape[1:]:
        raise ValueError(
            f"Dimension mismatch: difference {diff.shape}, AFTER {after.shape[1:]}"
        )
    if diff_profile["crs"] != after_profile["crs"]:
        raise ValueError("CRS mismatch between difference and AFTER raster.")
    if diff_profile["transform"] != after_profile["transform"]:
        raise ValueError("Transform mismatch between difference and AFTER raster.")

    valid_diff = np.isfinite(diff) & (diff != diff_nodata)
    valid_after = np.all(np.isfinite(after) & (after != after_nodata), axis=0)
    valid = valid_diff & valid_after

    values = diff[valid]
    if values.size == 0:
        raise ValueError("No common valid pixels available.")

    rgb = stretch_rgb(after, valid)

    print("=" * 65)
    print("EXPLORATORY CANDIDATE CHANGE MASKS")
    print("Shivamogga | 13 Jan 2016 – 20 Jan 2026")
    print("=" * 65)
    print("Valid pixels:", int(valid.sum()))
    print("Pixel size: 10 m x 10 m (0.01 hectare per pixel)")
    print("Thresholds are percentile-based screening values, not validated classes.")

    for percentile in PERCENTILES:
        threshold = float(np.percentile(values, percentile))
        mask = valid & (diff >= threshold)
        count = int(mask.sum())
        area_ha = count * 0.01
        percentage = count / int(valid.sum()) * 100

        tag = f"p{percentile}"
        mask_path = OUTPUT_DIR / f"candidate_change_mask_{tag}.tif"
        overlay_path = OUTPUT_DIR / f"candidate_change_overlay_{tag}.png"

        out_profile = diff_profile.copy()
        out_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=255,
            compress="deflate",
        )
        out_mask = np.full(mask.shape, 255, dtype=np.uint8)
        out_mask[valid] = mask[valid].astype(np.uint8)

        with rasterio.open(mask_path, "w", **out_profile) as dst:
            dst.write(out_mask, 1)
            dst.set_band_description(1, f"Candidate_change_{percentile}th_percentile")

        overlay = rgb.copy()
        # Red overlay for candidate pixels.
        overlay[mask] = (
            0.45 * overlay[mask] + 0.55 * np.array([255, 0, 0])
        ).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(rgb)
        axes[0].set_title("AFTER image (20 Jan 2026)")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(
            f"Candidate changes: {percentile}th percentile\n"
            f"Threshold = {threshold:.6f}"
        )
        axes[1].axis("off")

        fig.suptitle(
            f"Exploratory Sentinel-2 Candidate Change Map — {percentile}th percentile",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(overlay_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        print(f"\n{percentile}th percentile threshold: {threshold:.6f}")
        print("Candidate pixels:", count)
        print("Candidate area:", f"{area_ha:.2f} hectares")
        print("Study area percentage:", f"{percentage:.2f}%")
        print("Mask:", mask_path)
        print("Overlay:", overlay_path)

    print("\nCompleted. These masks identify spectral outliers only.")
    print("They are not ground-truth labels or confirmed human-induced changes.")
    print("Original aligned rasters and spectral difference rasters were not modified.")


if __name__ == "__main__":
    main()
