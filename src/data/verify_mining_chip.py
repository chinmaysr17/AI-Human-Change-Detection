
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/sentinel2/mining"
MASK = ROOT / "dataset/Mining/chips/mining_mask_2805.tif"

bands = {}

for band in ["B04", "B03", "B02"]:
    path = DATA / f"mining_2805_{band}.tif"

    with rasterio.open(path) as src:
        bands[band] = src.read(1).astype(np.float32)

with rasterio.open(MASK) as src:
    mask = src.read(1)

rgb = np.stack(
    [bands["B04"], bands["B03"], bands["B02"]],
    axis=-1
)

# Percentile contrast enhancement for display.
low, high = np.percentile(rgb, [2, 98])
rgb = np.clip((rgb - low) / (high - low + 1e-8), 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Mining Ground-Truth Mask")

axes[2].imshow(rgb)
axes[2].imshow(mask, cmap="autumn", alpha=0.4)
axes[2].set_title("Mining Polygon Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output = ROOT / "dataset/Mining/chips/mining_verification.png"
plt.savefig(output, dpi=150)
print("Saved visual verification:", output)

plt.show()
