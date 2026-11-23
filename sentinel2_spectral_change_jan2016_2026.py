"""
Sentinel-2 spectral change analysis for the Shivamogga January 2016–2026 pair.

Inputs:
  data/sentinel2/aligned/before_aligned.tif
  data/sentinel2/aligned/after_aligned.tif

The inputs must be aligned, 3-band Float32 GeoTIFFs with matching CRS,
transform, dimensions, and NoData metadata. This script does not modify them.
"""

from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
BEFORE = BASE_DIR / "data" / "sentinel2" / "aligned" / "before_aligned.tif"
AFTER = BASE_DIR / "data" / "sentinel2" / "aligned" / "after_aligned.tif"

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "sentinel2"
    / "analyses"
    / "shivamogga_20160113_20260120_spectral"
)

NODATA_FALLBACK = -9999.0
BAND_NAMES = ["B04_Red", "B03_Green", "B02_Blue"]


def read_raster(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input raster not found: {path}")

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else NODATA_FALLBACK
        descriptions = src.descriptions

    return data, profile, nodata, descriptions


def main():
    print("=" * 68)
    print("SENTINEL-2 SPECTRAL CHANGE ANALYSIS")
    print("Shivamogga | 13 January 2016 – 20 January 2026")
    print("=" * 68)

    before, before_profile, before_nodata, before_desc = read_raster(BEFORE)
    after, after_profile, after_nodata, after_desc = read_raster(AFTER)

    # Validate image compatibility before calculating differences.
    if before.shape != after.shape:
        raise ValueError(
            f"Raster dimensions/bands differ: BEFORE {before.shape}, AFTER {after.shape}"
        )
    if before.shape[0] != 3:
        raise ValueError(f"Expected 3 bands (B04, B03, B02), found {before.shape[0]}")
    if before_profile["crs"] != after_profile["crs"]:
        raise ValueError("CRS mismatch between BEFORE and AFTER rasters.")
    if before_profile["transform"] != after_profile["transform"]:
        raise ValueError("Transform mismatch: rasters are not on the same grid.")
    if before_profile["width"] != after_profile["width"] or before_profile["height"] != after_profile["height"]:
        raise ValueError("Raster width/height mismatch.")

    # A pixel is valid only when every band on both dates is finite and not NoData.
    valid_before = np.all(np.isfinite(before) & (before != before_nodata), axis=0)
    valid_after = np.all(np.isfinite(after) & (after != after_nodata), axis=0)
    common_valid = valid_before & valid_after

    valid_count = int(common_valid.sum())
    total_count = int(common_valid.size)
    if valid_count == 0:
        raise ValueError("No common valid pixels found; cannot calculate spectral differences.")

    # Calculate per-band absolute reflectance differences and mean across 3 bands.
    difference = np.full(before.shape, NODATA_FALLBACK, dtype=np.float32)
    raw_difference = np.abs(after - before)
    difference[:, common_valid] = raw_difference[:, common_valid]

    mean_difference = np.full(common_valid.shape, NODATA_FALLBACK, dtype=np.float32)
    mean_difference[common_valid] = np.mean(raw_difference[:, common_valid], axis=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    diff_profile = before_profile.copy()
    diff_profile.update(
        driver="GTiff",
        count=3,
        dtype="float32",
        nodata=NODATA_FALLBACK,
        compress="deflate",
    )
    diff_path = OUTPUT_DIR / "sentinel2_3band_absolute_difference.tif"
    with rasterio.open(diff_path, "w", **diff_profile) as dst:
        dst.write(difference)
        for idx, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(idx, f"Absolute_Difference_{name}")

    mean_profile = before_profile.copy()
    mean_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=NODATA_FALLBACK,
        compress="deflate",
    )
    mean_path = OUTPUT_DIR / "sentinel2_mean_absolute_difference.tif"
    with rasterio.open(mean_path, "w", **mean_profile) as dst:
        dst.write(mean_difference, 1)
        dst.set_band_description(1, "Mean_Absolute_Reflectance_Difference")

    values = mean_difference[common_valid]
    vmin, vmax = np.percentile(values, [2, 98])
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    preview_path = OUTPUT_DIR / "sentinel2_spectral_difference_preview.png"
    plt.figure(figsize=(10, 8))
    masked = np.ma.masked_where(~common_valid, mean_difference)
    plt.imshow(masked, cmap="inferno", vmin=float(vmin), vmax=float(vmax))
    plt.colorbar(label="Mean absolute reflectance difference")
    plt.title("Sentinel-2 Mean Spectral Difference\n13 Jan 2016 – 20 Jan 2026")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(preview_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nINPUT VALIDATION")
    print("BEFORE:", BEFORE)
    print("AFTER :", AFTER)
    print("CRS:", before_profile["crs"])
    print("Dimensions:", before_profile["width"], "x", before_profile["height"])
    print("Bands:", ", ".join(BAND_NAMES))
    print("BEFORE NoData:", before_nodata)
    print("AFTER NoData :", after_nodata)
    print("Common valid pixels:", valid_count, "/", total_count)
    print("Common valid percentage:", f"{valid_count / total_count * 100:.2f}%")

    print("\nSPECTRAL DIFFERENCE STATISTICS (valid pixels only)")
    print("Mean:", f"{float(np.mean(values)):.6f}")
    print("Median:", f"{float(np.median(values)):.6f}")
    print("95th percentile:", f"{float(np.percentile(values, 95)):.6f}")
    print("Maximum:", f"{float(np.max(values)):.6f}")

    print("\nOUTPUTS")
    print("3-band difference GeoTIFF:", diff_path)
    print("Mean difference GeoTIFF :", mean_path)
    print("Preview PNG              :", preview_path)
    print("\nAnalysis completed. Original input rasters were not modified.")


if __name__ == "__main__":
    main()
