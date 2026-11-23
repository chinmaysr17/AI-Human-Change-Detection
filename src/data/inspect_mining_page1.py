
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset/Mining/training_chips_30"

tile_ids = [783, 926, 773, 2656, 930, 2200, 2028]

for tile_id in tile_ids:
    image_path = DATA / f"mining_{tile_id}.npy"
    mask_path = DATA / f"mask_{tile_id}.npy"

    if not image_path.exists() or not mask_path.exists():
        print(f"Skipping tile {tile_id}: files missing")
        continue

    image = np.load(image_path)
    mask = np.load(mask_path)

    # Convert Sentinel-2 bands to RGB.
    rgb = image[:, :, [2, 1, 0]].astype(np.float32)

    low, high = np.percentile(rgb, [2, 98])
    rgb = np.clip(
        (rgb - low) / (high - low + 1e-8),
        0,
        1
    )

    coverage = np.mean(mask > 0) * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title(f"Tile {tile_id}: Satellite RGB")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-truth mining mask")

    axes[2].imshow(rgb)
    axes[2].imshow(mask, cmap="autumn", alpha=0.45)
    axes[2].set_title(
        f"Mining overlay | Coverage: {coverage:.2f}%"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = DATA / f"detail_{tile_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

print("\nIndividual mining inspection images generated.")
