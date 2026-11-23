
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "dataset/Mining/splits"

split_names = ["train", "validation", "test"]

all_ids = {}
errors = []

print("\nMINING DATASET SPLIT VERIFICATION")
print("---------------------------------")

for split in split_names:

    folder = SPLIT_DIR / split

    images = {
        p.stem.replace("mining_", "")
        for p in folder.glob("mining_*.npy")
    }

    masks = {
        p.stem.replace("mask_", "")
        for p in folder.glob("mask_*.npy")
    }

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Masks :", len(masks))

    if images != masks:
        errors.append(f"{split}: image-mask mismatch")

    all_ids[split] = images

# Check for duplicate tile IDs across splits
for i, first in enumerate(split_names):
    for second in split_names[i + 1:]:

        overlap = all_ids[first] & all_ids[second]

        if overlap:
            errors.append(
                f"Data leakage between {first} and {second}: {overlap}"
            )

# Verify split manifest
manifest_path = SPLIT_DIR / "split_manifest.csv"

with open(manifest_path, newline="", encoding="utf-8") as file:
    manifest = list(csv.DictReader(file))

print("\nMANIFEST RECORDS:", len(manifest))
print("UNIQUE TILE IDS:", len({
    row["tile_id"] for row in manifest
}))

if len(manifest) != 24:
    errors.append("Manifest does not contain 24 records")

if len({row["tile_id"] for row in manifest}) != 24:
    errors.append("Duplicate tile IDs in manifest")

if errors:
    print("\nVERIFICATION FAILED")
    for error in errors:
        print("-", error)
else:
    print("\nALL CHECKS PASSED")
    print("No image-mask mismatches.")
    print("No duplicate tiles across splits.")
