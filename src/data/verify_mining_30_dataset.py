
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset/Mining/training_chips_30"

with open(DATA / "extraction_report.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

successful = [
    int(row["tile_id"])
    for row in rows
    if row["status"] == "success"
]

# Seven samples per verification page.
for page in range(4):
    tile_batch = successful[page * 7:(page + 1) * 7]

    if not tile_batch:
        continue

    fig, axes = plt.subplots(
        len(tile_batch), 3,
        figsize=(15, len(tile_batch) * 4)
    )

    if len(tile_batch) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, tile_id in enumerate(tile_batch):
        image = np.load(DATA / f"mining_{tile_id}.npy")
        mask = np.load(DATA / f"mask_{tile_id}.npy")

        rgb = image[:, :, [2, 1, 0]].astype(np.float32)
        low, high = np.percentile(rgb, [2, 98])
        rgb = np.clip(
            (rgb - low) / (high - low + 1e-8), 0, 1
        )

        coverage = mask.mean() * 100

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Tile {tile_id} - RGB")

        axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Mining Mask: {coverage:.2f}%")

        axes[i, 2].imshow(rgb)
        axes[i, 2].imshow(mask, cmap="autumn", alpha=0.4)
        axes[i, 2].set_title("Mining Polygon Overlay")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()

    output = DATA / f"mining_verification_page_{page + 1}.png"
    plt.savefig(output, dpi=130, bbox_inches="tight")
    plt.close()

    print("Saved:", output)

print("\nAll verification pages generated.")
