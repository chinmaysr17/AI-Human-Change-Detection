
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset/Mining/training_chips_30"

tile_ids = [783, 926, 773, 2656, 930, 2200, 2028]

for tile_id in tile_ids:
    image = np.load(DATA / f"mining_{tile_id}.npy")
    mask = np.load(DATA / f"mask_{tile_id}.npy")

    rgb = image[:, :, [2, 1, 0]].astype(np.float32)

    low, high = np.percentile(rgb, [2, 98])
    rgb = np.clip(
        (rgb - low) / (high - low + 1e-8), 0, 1
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title(f"Tile {tile_id}: Satellite RGB")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-truth mining mask")

    axes[2].imshow(rgb)
    axes[2].imshow(mask, cmap="autumn", alpha=0.45)
    axes[2].set_title("Mining overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output = DATA / f"detail_{tile_id}.png"
    plt.savefig(output, dpi=150)
    plt.close()

    print("Saved:", output)

print("Page 1 detailed inspections completed.")

