
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset/Mining/training_chips"

tile_ids = [1612, 2352, 1167, 1107, 1572]

fig, axes = plt.subplots(5, 3, figsize=(15, 22))

for i, tile_id in enumerate(tile_ids):
    image = np.load(DATA / f"mining_{tile_id}.npy")
    mask = np.load(DATA / f"mask_{tile_id}.npy")

    # RGB = B04, B03, B02
    rgb = image[:, :, [2, 1, 0]].astype(np.float32)

    low, high = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - low) / (high - low + 1e-8), 0, 1)

    coverage = mask.mean() * 100

    axes[i, 0].imshow(rgb)
    axes[i, 0].set_title(f"Tile {tile_id}: RGB")

    axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[i, 1].set_title(f"Mining Mask: {coverage:.2f}%")

    axes[i, 2].imshow(rgb)
    axes[i, 2].imshow(mask, cmap="autumn", alpha=0.4)
    axes[i, 2].set_title("Mining Overlay")

    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()

output = DATA / "training_dataset_verification.png"
plt.savefig(output, dpi=150, bbox_inches="tight")

print("Verification image saved:", output)
plt.show()
