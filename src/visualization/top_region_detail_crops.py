import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import ndimage

BASE = "data/sentinel2/processed"
MASK_DIR = os.path.join(BASE, "masked")
CHANGE_DIR = os.path.join(BASE, "change_analysis")

BEFORE = os.path.join(
    MASK_DIR, "sentinel2_20260927_shivamogga_10km_masked.tif"
)

AFTER = os.path.join(
    MASK_DIR, "sentinel2_20261027_shivamogga_10km_masked.tif"
)

CANDIDATE = os.path.join(
    CHANGE_DIR, "candidate_change_map.tif"
)

OUTPUT_DIR = os.path.join(CHANGE_DIR, "region_details")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PADDING = 30

with rasterio.open(BEFORE) as src:
    before = src.read()

with rasterio.open(AFTER) as src:
    after = src.read()

with rasterio.open(CANDIDATE) as src:
    candidate = src.read(1)

valid = (
    np.all(before > 0, axis=0)
    & np.all(after > 0, axis=0)
)

candidate_mask = (candidate == 1) & valid

labels, count = ndimage.label(
    candidate_mask,
    structure=np.ones((3, 3), dtype=bool)
)

sizes = np.bincount(labels.ravel())[1:]
top_ids = np.argsort(sizes)[::-1][:10] + 1


def create_rgb(data, mask):
    rgb = np.stack(
        [data[2], data[1], data[0]],
        axis=-1
    ).astype(np.float32)

    for band in range(3):
        values = rgb[:, :, band][mask]

        low, high = np.percentile(values, [2, 98])

        rgb[:, :, band] = np.clip(
            (rgb[:, :, band] - low) /
            max(high - low, 1),
            0,
            1
        )

    return rgb


rgb_before = create_rgb(before, valid)
rgb_after = create_rgb(after, valid)

print("Generating detailed candidate-region crops...")

for region_id in top_ids:

    rows, cols = np.where(labels == region_id)

    if len(rows) == 0:
        continue

    # Region bounding box with padding
    r_min = max(0, rows.min() - PADDING)
    r_max = min(labels.shape[0], rows.max() + PADDING + 1)

    c_min = max(0, cols.min() - PADDING)
    c_max = min(labels.shape[1], cols.max() + PADDING + 1)

    # Extract image crops
    before_crop = rgb_before[r_min:r_max, c_min:c_max]
    after_crop = rgb_after[r_min:r_max, c_min:c_max]

    # Exact candidate pixels for this region
    region_mask = (
        labels[r_min:r_max, c_min:c_max] == region_id
    )

    # Red overlay on exact candidate pixels
    overlay = np.zeros(
        (*region_mask.shape, 4),
        dtype=np.float32
    )

    overlay[region_mask] = [1, 0, 0, 0.75]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, image, title in [
        (axes[0], before_crop, "September 27, 2026"),
        (axes[1], after_crop, "October 27, 2026")
    ]:
        ax.imshow(image)
        ax.imshow(overlay)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"Candidate Region {region_id} | "
        f"{len(rows)} pixels | {len(rows) * 0.01:.2f} ha",
        fontsize=13
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"region_{region_id}_comparison.png"
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print("Saved:", output_path)

print("\nAll 10 region comparisons generated.")
print("Output folder:", OUTPUT_DIR)