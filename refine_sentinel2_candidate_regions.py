"""
Spatially refine and group exploratory Sentinel-2 candidate changes.

Inputs are the aligned January 2016/2026 GeoTIFFs and previously generated
mean spectral difference raster. Produces filtered masks, connected-region
rasters, CSV region statistics, and BEFORE/AFTER/change comparison figures.

This is exploratory screening, not validated human-induced change detection.
"""

from pathlib import Path
import csv

import numpy as np
import rasterio
from rasterio.warp import transform as transform_coords
from scipy import ndimage
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "sentinel2"

BEFORE_PATH = DATA_DIR / "aligned" / "before_aligned.tif"
AFTER_PATH = DATA_DIR / "aligned" / "after_aligned.tif"
DIFF_PATH = (
    DATA_DIR / "analyses" / "shivamogga_20160113_20260120_spectral"
    / "sentinel2_mean_absolute_difference.tif"
)
CANDIDATE_DIR = DATA_DIR / "analyses" / "shivamogga_20160113_20260120_candidates"

OUTPUT_DIR = (
    DATA_DIR / "analyses" / "shivamogga_20160113_20260120_spatial_refinement"
)

# Remove tiny isolated regions smaller than 9 pixels = 0.09 ha at 10 m.
MIN_REGION_PIXELS = 9

# 8-connected neighbourhood: diagonal neighbours belong to the same region.
STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def read_raster(path, indexes=None):
    if not path.exists():
        raise FileNotFoundError(f"Required raster not found: {path}")
    with rasterio.open(path) as src:
        data = src.read(indexes=indexes).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        descriptions = src.descriptions
    return data, profile, nodata, descriptions


def make_rgb(bands, nodata):
    # Inputs are ordered B04, B03, B02. Display RGB is B02, B03, B04.
    valid = np.all(np.isfinite(bands), axis=0)
    if nodata is not None:
        valid &= np.all(bands != nodata, axis=0)

    rgb_source = np.stack([bands[2], bands[1], bands[0]], axis=-1)
    rgb = np.zeros(rgb_source.shape, dtype=np.uint8)

    for channel in range(3):
        vals = rgb_source[:, :, channel][valid]
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        scaled = np.clip((rgb_source[:, :, channel] - lo) / (hi - lo), 0, 1)
        rgb[:, :, channel] = (scaled * 255).astype(np.uint8)

    return rgb, valid


def save_mask(path, mask, valid, profile, description):
    output = np.full(mask.shape, 255, dtype=np.uint8)
    output[valid] = mask[valid].astype(np.uint8)

    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(output, 1)
        dst.set_band_description(1, description)


def region_statistics(labels, count, profile, csv_path, threshold_name):
    pixel_area_ha = (
        abs(profile["transform"].a * profile["transform"].e
            - profile["transform"].b * profile["transform"].d) / 10000.0
    )
    transform = profile["transform"]
    rows = []

    for region_id, region_slice in enumerate(
        ndimage.find_objects(labels), start=1
    ):
        if region_slice is None:
            continue

        local = labels[region_slice] == region_id
        pixel_count = int(local.sum())
        if pixel_count == 0:
            continue

        local_rows, local_cols = np.nonzero(local)
        row_c = float(region_slice[0].start + local_rows.mean())
        col_c = float(region_slice[1].start + local_cols.mean())

        x, y = transform * (col_c + 0.5, row_c + 0.5)
        lon, lat = (None, None)
        if profile.get("crs") is not None:
            try:
                lon_list, lat_list = transform_coords(
                    profile["crs"], "EPSG:4326", [x], [y]
                )
                lon, lat = float(lon_list[0]), float(lat_list[0])
            except Exception:
                pass

        rows.append({
            "threshold": threshold_name,
            "region_id": region_id,
            "pixel_count": pixel_count,
            "area_hectares": round(pixel_count * pixel_area_ha, 4),
            "centroid_x": round(float(x), 3),
            "centroid_y": round(float(y), 3),
            "longitude": round(lon, 7) if lon is not None else "",
            "latitude": round(lat, 7) if lat is not None else "",
        })

    fieldnames = [
        "threshold", "region_id", "pixel_count", "area_hectares",
        "centroid_x", "centroid_y", "longitude", "latitude"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows, pixel_area_ha


def main():
    print("=" * 68)
    print("SENTINEL-2 SPATIAL REFINEMENT AND REGION GROUPING")
    print("Shivamogga | 13 Jan 2016 – 20 Jan 2026")
    print("=" * 68)

    before, before_profile, before_nodata, _ = read_raster(BEFORE_PATH)
    after, after_profile, after_nodata, _ = read_raster(AFTER_PATH)
    diff_data, diff_profile, diff_nodata, _ = read_raster(DIFF_PATH)
    diff = diff_data[0]

    for label, profile in [("AFTER", after_profile), ("DIFFERENCE", diff_profile)]:
        if profile["crs"] != before_profile["crs"]:
            raise ValueError(f"{label} CRS does not match BEFORE.")
        if profile["transform"] != before_profile["transform"]:
            raise ValueError(f"{label} transform does not match BEFORE.")
        if profile["width"] != before_profile["width"] or profile["height"] != before_profile["height"]:
            raise ValueError(f"{label} dimensions do not match BEFORE.")

    rgb_before, valid_before = make_rgb(before, before_nodata)
    rgb_after, valid_after = make_rgb(after, after_nodata)
    valid_diff = np.isfinite(diff)
    if diff_nodata is not None:
        valid_diff &= diff != diff_nodata
    valid = valid_before & valid_after & valid_diff

    if not np.any(valid):
        raise ValueError("No common valid pixels found.")

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_valid = int(valid.sum())
    print("CRS:", before_profile["crs"])
    print("Dimensions:", before_profile["width"], "x", before_profile["height"])
    print("Common valid pixels:", total_valid)
    print("Minimum region size:", MIN_REGION_PIXELS, "pixels")
    print("Connectivity: 8-neighbour")

    combined_csv = OUTPUT_DIR / "all_candidate_region_statistics.csv"
    all_rows = []

    for percentile in (95, 98):
        tag = f"p{percentile}"
        candidate_path = CANDIDATE_DIR / f"candidate_change_mask_{tag}.tif"
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate mask not found: {candidate_path}")

        with rasterio.open(candidate_path) as src:
            original = src.read(1)
            candidate_nodata = src.nodata

        candidate_valid = original != candidate_nodata if candidate_nodata is not None else np.ones(original.shape, dtype=bool)
        candidate_valid &= valid
        raw_mask = (original == 1) & candidate_valid

        # 3x3 median filter suppresses isolated speckles while retaining
        # broad spatial structures. Re-apply validity after filtering.
        filtered = ndimage.median_filter(raw_mask.astype(np.uint8), size=3) > 0
        filtered &= valid

        # Group adjacent pixels and remove tiny connected components.
        labels, n_labels = ndimage.label(filtered, structure=STRUCTURE)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= MIN_REGION_PIXELS
        keep[0] = False
        cleaned = keep[labels] & valid

        final_labels, final_count = ndimage.label(cleaned, structure=STRUCTURE)
        final_sizes = np.bincount(final_labels.ravel())
        final_sizes[0] = 0
        if final_count:
            # Re-label in descending area order for readable region IDs.
            order = np.argsort(final_sizes[1:])[::-1] + 1
            remap = np.zeros(final_count + 1, dtype=np.int32)
            for new_id, old_id in enumerate(order, start=1):
                remap[old_id] = new_id
            final_labels = remap[final_labels]
            final_count = len(order)

        cleaned = final_labels > 0

        filtered_path = OUTPUT_DIR / f"filtered_candidate_mask_{tag}.tif"
        regions_path = OUTPUT_DIR / f"connected_change_regions_{tag}.tif"
        stats_path = OUTPUT_DIR / f"region_statistics_{tag}.csv"
        figure_path = OUTPUT_DIR / f"final_comparison_{tag}.png"

        save_mask(
            filtered_path, cleaned, valid, before_profile,
            f"Filtered_candidate_change_{tag}"
        )

        region_out = final_labels.astype(np.uint32)
        region_profile = before_profile.copy()
        region_profile.update(
            driver="GTiff", count=1, dtype="uint32", nodata=0,
            compress="deflate"
        )
        with rasterio.open(regions_path, "w", **region_profile) as dst:
            dst.write(region_out, 1)
            dst.set_band_description(1, f"Connected_candidate_regions_{tag}")

        rows, pixel_area_ha = region_statistics(
            final_labels, final_count, before_profile, stats_path, tag
        )
        all_rows.extend(rows)

        candidate_pixels = int(cleaned.sum())
        area_ha = candidate_pixels * pixel_area_ha
        percentage = candidate_pixels / total_valid * 100

        # Region IDs are visualized with a categorical palette.
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        axes[0].imshow(rgb_before)
        axes[0].set_title("BEFORE — 13 Jan 2016")
        axes[1].imshow(rgb_after)
        axes[1].set_title("AFTER — 20 Jan 2026")

        overlay = rgb_after.copy()
        overlay[cleaned] = (
            0.42 * overlay[cleaned] + 0.58 * np.array([255, 0, 0])
        ).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Filtered candidate changes — {tag.upper()}\n"
            f"{candidate_pixels:,} pixels | {area_ha:.2f} ha | "
            f"{final_count} regions"
        )

        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"Shivamogga Sentinel-2 Candidate Change Screening ({tag.upper()})",
            fontsize=15,
        )
        fig.tight_layout()
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        print(f"\n{tag.upper()} RESULTS")
        print("Original candidate pixels:", int(raw_mask.sum()))
        print("Pixels after spatial filtering:", candidate_pixels)
        print("Candidate area:", f"{area_ha:.2f} hectares")
        print("Share of valid study area:", f"{percentage:.2f}%")
        print("Connected regions:", final_count)
        print("Largest region:", f"{int(final_sizes.max()) if final_count else 0} pixels")
        print("Filtered mask:", filtered_path)
        print("Region raster:", regions_path)
        print("Region CSV:", stats_path)
        print("Comparison figure:", figure_path)

    with open(combined_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "threshold", "region_id", "pixel_count", "area_hectares",
            "centroid_x", "centroid_y", "longitude", "latitude"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nCOMBINED REGION TABLE:", combined_csv)
    print("\nCompleted. These remain exploratory spectral-change candidates.")
    print("No CNN inference, ground-truth validation, or human-activity classification was performed.")
    print("Original aligned imagery and previous analysis outputs were not modified.")


if __name__ == "__main__":
    main()
