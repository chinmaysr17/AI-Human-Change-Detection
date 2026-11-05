
from pathlib import Path
import csv
import random

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset/Mining/component_chips"

with open(DATA / "component_audit.csv", newline="") as f:
    rows = list(csv.DictReader(f))

candidates = [
    row["sample"] for row in rows
    if row["status"] == "CANDIDATE"
]

random.seed(51)
selected = random.sample(candidates, min(12, len(candidates)))

fig, axes = plt.subplots(12, 3, figsize=(14, 36))

for i, sample in enumerate(selected):
    image = np.load(DATA / f"{sample}_image.npy")
    mask = np.load(DATA / f"{sample}_mask.npy")

    rgb = image[:, :, [2, 1, 0]].astype(np.float32)

    low, high = np.percentile(rgb, [2, 98])
    rgb = np.clip(
        (rgb - low) / (high - low + 1e-8), 0, 1
    )

    coverage = mask.mean() * 100

    axes[i, 0].imshow(rgb)
    axes[i, 0].set_title(f"{sample} - RGB")

    axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[i, 1].set_title(f"Mask - {coverage:.2f}%")

    axes[i, 2].imshow(rgb)
    axes[i, 2].imshow(mask, cmap="autumn", alpha=0.45)
    axes[i, 2].set_title("Overlay")

    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()

output = DATA / "mining_candidate_review.png"
plt.savefig(output, dpi=120, bbox_inches="tight")
plt.close()

print("Selected samples:", len(selected))
print("Saved:", output)
