
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset/Mining/component_chips"

images = sorted(DATA.glob("*_image.npy"))

results = []

for image_path in images:
    mask_path = image_path.with_name(
        image_path.name.replace("_image.npy", "_mask.npy")
    )

    if not mask_path.exists():
        continue

    image = np.load(image_path)
    mask = np.load(mask_path)

    coverage = float(mask.mean() * 100)
    valid_pixels = np.any(image != 0, axis=-1)
    valid_percent = float(valid_pixels.mean() * 100)

    results.append({
        "sample": image_path.stem.replace("_image", ""),
        "coverage_percent": round(coverage, 2),
        "valid_image_percent": round(valid_percent, 2),
        "image_shape": str(image.shape),
        "mask_shape": str(mask.shape),
        "status": (
            "REVIEW"
            if coverage < 1 or coverage > 95 or valid_percent < 95
            else "CANDIDATE"
        )
    })

output = DATA / "component_audit.csv"

with open(output, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nMINING COMPONENT AUDIT")
print("-" * 45)
print("Total image-mask pairs:", len(results))
print("Candidates:", sum(r["status"] == "CANDIDATE" for r in results))
print("Require review:", sum(r["status"] == "REVIEW" for r in results))
print("Audit saved:", output)
