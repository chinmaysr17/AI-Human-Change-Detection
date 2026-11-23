"""
Extract and visualize the 10 largest P95 candidate regions.

Inputs:
  data/sentinel2/aligned/before_aligned.tif
  data/sentinel2/aligned/after_aligned.tif
  data/sentinel2/analyses/shivamogga_20160113_20260120_spatial_refinement/
      connected_change_regions_p95.tif
  data/sentinel2/analyses/shivamogga_20160113_20260120_spatial_refinement/
      region_statistics_p95.csv

Creates individual crop panels, a contact sheet, and a top-10 CSV.
No source rasters are modified.
"""

from pathlib import Path
import csv

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "sentinel2"

BEFORE_PATH = DATA_DIR / "aligned" / "before_aligned.tif"
AFTER_PATH = DATA_DIR / "aligned" / "after_aligned.tif"

REFINEMENT_DIR = (
    DATA_DIR / "analyses" / "shivamogga_20160113_20260120_spatial_refinement"
)
REGIONS_PATH = REFINEMENT_DIR / "connected_change_regions_p95.tif"
STATS_PATH = REFINEMENT_DIR / "region_statistics_p95.csv"

OUTPUT_DIR = (
    DATA_DIR / "analyses" / "shivamogga_20160113_20260120_top10_regions"
)

TOP_N = 10
PADDING_PIXELS = 40


def load_rgb_pair(before_path, after_path):
    for path in (before_path, after_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    with rasterio.open(before_path) as bsrc, rasterio.open(after_path) as asrc:
        if (
            bsrc.crs != asrc.crs
            or bsrc.transform != asrc.transform
            or bsrc.width != asrc.width
            or bsrc.height != asrc.height
        ):
            raise ValueError("BEFORE and AFTER rasters are not aligned.")

        before = bsrc.read().astype(np.float32)
        after = asrc.read().astype(np.float32)
        profile = bsrc.profile.copy()
        nodata_before = bsrc.nodata
        nodata_after = asrc.nodata

    valid_before = np.all(np.isfinite(before), axis=0)
    valid_after = np.all(np.isfinite(after), axis=0)
    if nodata_before is not None:
        valid_before &= np.all(before != nodata_before, axis=0)
    if nodata_after is not None:
        valid_after &= np.all(after != nodata_after, axis=0)
    valid = valid_before & valid_after

    if not np.any(valid):
        raise ValueError("No common valid pixels.")

    # Shared stretch across both dates, separately for each displayed channel.
    # Input bands: B04, B03, B02; output display order: B02, B03, B04.
    before_rgb_src = np.stack([before[2], before[1], before[0]], axis=-1)
    after_rgb_src = np.stack([after[2], after[1], after[0]], axis=-1)

    before_rgb = np.zeros(before_rgb_src.shape, dtype=np.uint8)
    after_rgb = np.zeros(after_rgb_src.shape, dtype=np.uint8)

    for channel in range(3):
        combined = np.concatenate([
            before_rgb_src[:, :, channel][valid],
            after_rgb_src[:, :, channel][valid],
        ])
        low, high = np.percentile(combined, [2, 98])
        if high <= low:
            high = low + 1e-6

        before_scaled = np.clip(
            (before_rgb_src[:, :, channel] - low) / (high - low), 0, 1
        )
        after_scaled = np.clip(
            (after_rgb_src[:, :, channel] - low) / (high - low), 0, 1
        )
        before_rgb[:, :, channel] = (before_scaled * 255).astype(np.uint8)
        after_rgb[:, :, channel] = (after_scaled * 255).astype(np.uint8)

    return before_rgb, after_rgb, profile, valid


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    before_rgb, after_rgb, profile, valid = load_rgb_pair(
        BEFORE_PATH, AFTER_PATH
    )

    if not REGIONS_PATH.exists():
        raise FileNotFoundError(f"Region raster not found: {REGIONS_PATH}")
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Region statistics CSV not found: {STATS_PATH}")

    with rasterio.open(REGIONS_PATH) as src:
        labels = src.read(1)
        region_profile = src.profile.copy()

    if labels.shape != valid.shape:
        raise ValueError("Region raster dimensions do not match image rasters.")
    if region_profile["crs"] != profile["crs"] or region_profile["transform"] != profile["transform"]:
        raise ValueError("Region raster georeferencing does not match image rasters.")

    stats = pd.read_csv(STATS_PATH)
    top = stats.sort_values(
        ["pixel_count", "region_id"], ascending=[False, True]
    ).head(TOP_N).copy()

    if top.empty:
        raise ValueError("No candidate regions found in statistics CSV.")

    transform = profile["transform"]
    height, width = labels.shape
    summary_rows = []
    crop_paths = []
    overlay_thumbnails = []
    thumbnail_titles = []

    print("=" * 68)
    print("TOP 10 LARGEST P95 CANDIDATE REGIONS")
    print("Shivamogga | 13 Jan 2016 – 20 Jan 2026")
    print("=" * 68)
    print("Shared RGB stretch used for BEFORE and AFTER comparison.")
    print("Crop padding:", PADDING_PIXELS, "pixels around each region bounding box.")

    for rank, row in enumerate(top.itertuples(index=False), start=1):
        region_id = int(row.region_id)
        region_mask = labels == region_id
        rows, cols = np.where(region_mask)

        if rows.size == 0:
            continue

        r0 = max(0, int(rows.min()) - PADDING_PIXELS)
        r1 = min(height, int(rows.max()) + 1 + PADDING_PIXELS)
        c0 = max(0, int(cols.min()) - PADDING_PIXELS)
        c1 = min(width, int(cols.max()) + 1 + PADDING_PIXELS)

        before_crop = before_rgb[r0:r1, c0:c1]
        after_crop = after_rgb[r0:r1, c0:c1]
        mask_crop = region_mask[r0:r1, c0:c1] & valid[r0:r1, c0:c1]

        overlay = after_crop.copy()
        overlay[mask_crop] = (
            0.38 * overlay[mask_crop] + 0.62 * np.array([255, 0, 0])
        ).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(before_crop)
        axes[0].set_title("BEFORE — 13 Jan 2016")
        axes[1].imshow(after_crop)
        axes[1].set_title("AFTER — 20 Jan 2026")
        axes[2].imshow(overlay)
        axes[2].set_title("P95 candidate region (red)")

        for ax in axes:
            ax.axis("off")

        area_ha = float(row.area_hectares)
        pixels = int(row.pixel_count)
        fig.suptitle(
            f"Rank {rank} | Region {region_id} | {area_ha:.2f} ha | "
            f"{pixels} pixels | {float(row.latitude):.6f}, "
            f"{float(row.longitude):.6f}",
            fontsize=12,
        )
        fig.tight_layout()

        image_path = OUTPUT_DIR / f"top{rank:02d}_region_{region_id}_comparison.png"
        fig.savefig(image_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        crop_paths.append(image_path)
        overlay_thumbnails.append(overlay.copy())
        thumbnail_titles.append(f"Rank {rank} | Region {region_id} | {area_ha:.2f} ha")

        # Geographic bounds for the crop.
        left, top_y = transform * (c0, r0)
        right, bottom_y = transform * (c1, r1)

        summary_rows.append({
            "rank": rank,
            "region_id": region_id,
            "pixel_count": pixels,
            "area_hectares": area_ha,
            "centroid_latitude": float(row.latitude),
            "centroid_longitude": float(row.longitude),
            "crop_row_start": r0,
            "crop_row_end_exclusive": r1,
            "crop_col_start": c0,
            "crop_col_end_exclusive": c1,
            "crop_left_x": left,
            "crop_bottom_y": bottom_y,
            "crop_right_x": right,
            "crop_top_y": top_y,
            "comparison_image": image_path.name,
        })

        print(
            f"Rank {rank:02d} | Region {region_id:4d} | "
            f"{pixels:4d} pixels | {area_ha:6.2f} ha | "
            f"Lat {float(row.latitude):.6f} | "
            f"Lon {float(row.longitude):.6f}"
        )

    summary_path = OUTPUT_DIR / "top10_region_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Contact sheet of the 10 AFTER-image crops with candidate pixels in red.
    cols = 2
    rows_count = int(np.ceil(len(overlay_thumbnails) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(14, 5 * rows_count))
    axes = np.array(axes).reshape(rows_count, cols)

    for idx, ax in enumerate(axes.flat):
        if idx < len(overlay_thumbnails):
            ax.imshow(overlay_thumbnails[idx])
            ax.set_title(thumbnail_titles[idx])
        ax.axis("off")

    fig.suptitle(
        "Top 10 P95 Candidate Regions — AFTER Image Crops with Candidate Pixels",
        fontsize=15,
    )
    fig.tight_layout()
    contact_path = OUTPUT_DIR / "top10_regions_contact_sheet.png"
    fig.savefig(contact_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nOUTPUT DIRECTORY:", OUTPUT_DIR)
    print("Top 10 CSV:", summary_path)
    print("Top 10 contact sheet:", contact_path)
    print("Individual comparison images:", len(crop_paths))
    print("\nNote: These are candidate spectral-change regions, not confirmed")
    print("human-induced changes. Visual inspection is not ground-truth validation.")


if __name__ == "__main__":
    main()
