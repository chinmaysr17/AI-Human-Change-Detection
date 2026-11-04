import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.patches import Rectangle

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

OUTPUT = os.path.join(
    CHANGE_DIR, "top_region_comparison.png"
)

with rasterio.open(BEFORE) as src:
    before = src.read()
    transform = src.transform

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

def create_rgb(data, valid_mask):
    rgb = np.stack(
        [data[2], data[1], data[0]],
        axis=-1
    ).astype(np.float32)

    for band in range(3):
        values = rgb[:, :, band][valid_mask]
        low, high = np.percentile(values, [2, 98])
        rgb[:, :, band] = np.clip(
            (rgb[:, :, band] - low) / (high - low),
            0, 1
        )

    return rgb

rgb_before = create_rgb(before, valid)
rgb_after = create_rgb(after, valid)

fig, axes = plt.subplots(2, 1, figsize=(12, 16))

for ax, rgb, title in [
    (axes[0], rgb_before, "September 27, 2026"),
    (axes[1], rgb_after, "October 27, 2026")
]:
    ax.imshow(rgb)

    for region_id in top_ids:
        rows, cols = np.where(labels == region_id)

        if len(rows) == 0:
            continue

        x_min, x_max = cols.min(), cols.max()
        y_min, y_max = rows.min(), rows.max()

        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min + 1,
            y_max - y_min + 1,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)

        ax.text(
            x_min,
            max(0, y_min - 5),
            str(region_id),
            color="yellow",
            fontsize=9,
            weight="bold"
        )

    ax.set_title(title)
    ax.axis("off")

fig.suptitle(
    "Top 10 Candidate Regions — Before and After Comparison",
    fontsize=15
)

plt.tight_layout()

plt.savefig(OUTPUT, dpi=200, bbox_inches="tight")
plt.close()

print("Top region comparison generated.")
print("Top region IDs:", list(top_ids))
print("Output:", OUTPUT)