
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("dataset/Mining/training_chips_30")

TILE_IDS = [2120, 2987, 47, 783]

for tile_id in TILE_IDS:

    image = np.load(DATA_DIR / f"mining_{tile_id}.npy")
    mask = np.load(DATA_DIR / f"mask_{tile_id}.npy")

    # Sentinel-2 RGB: B04, B03, B02
    rgb = image[:, :, [2, 1, 0]].astype(np.float32)

    lower = np.percentile(rgb, 2)
    upper = np.percentile(rgb, 98)

    if upper > lower:
        rgb = np.clip((rgb - lower) / (upper - lower), 0, 1)
    else:
        rgb = np.zeros_like(rgb)

    coverage = np.mean(mask > 0) * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title(f"Satellite Image - {tile_id}")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Mining Mask - {coverage:.2f}%")

    axes[2].imshow(rgb)
    axes[2].imshow(
        mask > 0,
        cmap="Reds",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    axes[2].set_title("Mask Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = DATA_DIR / f"detail_{tile_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(
        f"Tile {tile_id}: "
        f"coverage={coverage:.2f}% | "
        f"unique mask values={np.unique(mask)} | "
        f"saved={output_path}"
    )

print("\nInspection images regenerated successfully.")
