
from pathlib import Path
import csv
import random
import shutil

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = ROOT / "dataset/Mining/training_ready"
OUTPUT_DIR = ROOT / "dataset/Mining/splits"

SEED = 42

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "validation"
TEST_DIR = OUTPUT_DIR / "test"

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD INCLUDED SAMPLES
# --------------------------------------------------

manifest_path = SOURCE_DIR / "training_manifest.csv"

with open(manifest_path, newline="", encoding="utf-8") as file:
    records = [
        row for row in csv.DictReader(file)
        if row["status"] == "INCLUDED"
    ]

# --------------------------------------------------
# GROUP BY MINING CATEGORY
# --------------------------------------------------

groups = {}

for row in records:
    category = (
        row["mining_type"],
        row["activity_type"]
    )

    groups.setdefault(category, []).append(row)

# --------------------------------------------------
# CREATE REPRODUCIBLE SPLITS
# --------------------------------------------------

rng = random.Random(SEED)

split_records = []

for category, samples in sorted(groups.items()):

    rng.shuffle(samples)

    if len(samples) < 3:
        raise ValueError(
            f"Not enough samples in category: {category}"
        )

    # One sample for validation and one for testing.
    validation_samples = samples[:1]
    test_samples = samples[1:2]
    training_samples = samples[2:]

    assignments = [
        (row, "train", TRAIN_DIR)
        for row in training_samples
    ] + [
        (row, "validation", VAL_DIR)
        for row in validation_samples
    ] + [
        (row, "test", TEST_DIR)
        for row in test_samples
    ]

    for row, split_name, destination in assignments:

        tile_id = row["tile_id"]

        for prefix in ["mining_", "mask_"]:
            filename = f"{prefix}{tile_id}.npy"

            source = SOURCE_DIR / filename
            target = destination / filename

            if not source.exists():
                raise FileNotFoundError(source)

            shutil.copy2(source, target)

        split_records.append({
            "tile_id": tile_id,
            "mining_type": row["mining_type"],
            "activity_type": row["activity_type"],
            "split": split_name
        })

# --------------------------------------------------
# SAVE SPLIT MANIFEST
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

split_manifest = OUTPUT_DIR / "split_manifest.csv"

with open(
    split_manifest,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fields = [
        "tile_id",
        "mining_type",
        "activity_type",
        "split"
    ]

    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(sorted(
        split_records,
        key=lambda row: int(row["tile_id"])
    ))

# --------------------------------------------------
# PRINT SUMMARY
# --------------------------------------------------

print("\nMINING DATASET SPLIT SUMMARY")
print("----------------------------")

for split_name in ["train", "validation", "test"]:

    count = sum(
        row["split"] == split_name
        for row in split_records
    )

    print(f"{split_name.capitalize():12}: {count} samples")

print("\nCATEGORY DISTRIBUTION")

for category in sorted(groups):

    print(f"\n{category[0]} - {category[1]}")

    for split_name in ["train", "validation", "test"]:

        count = sum(
            row["mining_type"] == category[0]
            and row["activity_type"] == category[1]
            and row["split"] == split_name
            for row in split_records
        )

        print(f"  {split_name}: {count}")

print("\nSplit manifest:", split_manifest)
print("Random seed:", SEED)
print("\nDataset splitting completed.")
