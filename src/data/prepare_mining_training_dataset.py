
from pathlib import Path
import csv
import shutil

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = ROOT / "dataset/Mining/training_chips_30"
OUTPUT_DIR = ROOT / "dataset/Mining/training_ready"

METADATA_FILE = ROOT / "dataset/Mining/mining_training_30.csv"
QUALITY_FILE = ROOT / "dataset/Mining/mining_dataset_quality_report.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Samples provisionally excluded from primary training
EXCLUDED = {
    "2120": "Broad polygon covers entire image chip",
    "2987": "Broad polygon covers almost entire image chip",
    "47": "Very low polygon overlap and boundary clipping",
    "783": "Very low polygon overlap and boundary clipping"
}

# --------------------------------------------------
# LOAD METADATA
# --------------------------------------------------

with open(METADATA_FILE, newline="", encoding="utf-8") as file:
    metadata = {
        row["tile_id"]: row
        for row in csv.DictReader(file)
    }

with open(QUALITY_FILE, newline="", encoding="utf-8") as file:
    quality = {
        row["tile_id"]: row
        for row in csv.DictReader(file)
    }

manifest = []

included = 0
excluded = 0

# --------------------------------------------------
# PREPARE TRAINING DATASET
# --------------------------------------------------

for tile_id, info in metadata.items():

    image_path = SOURCE_DIR / f"mining_{tile_id}.npy"
    mask_path = SOURCE_DIR / f"mask_{tile_id}.npy"

    if tile_id in EXCLUDED:
        status = "EXCLUDED"
        reason = EXCLUDED[tile_id]

    elif not image_path.exists() or not mask_path.exists():
        status = "EXCLUDED"
        reason = "Missing image or mask"

    elif tile_id not in quality:
        status = "EXCLUDED"
        reason = "Missing quality validation record"

    elif quality[tile_id]["status"] != "PASS":
        status = "EXCLUDED"
        reason = "Did not pass technical quality validation"

    else:
        status = "INCLUDED"
        reason = "Passed technical checks; pending model-level validation"

    if status == "INCLUDED":

        shutil.copy2(
            image_path,
            OUTPUT_DIR / image_path.name
        )

        shutil.copy2(
            mask_path,
            OUTPUT_DIR / mask_path.name
        )

        included += 1

    else:
        excluded += 1

    manifest.append({
        "tile_id": tile_id,
        "mining_type": info["mining_type"],
        "activity_type": info["activity_type"],
        "coverage_percent": quality.get(
            tile_id, {}
        ).get("coverage_percent", ""),
        "status": status,
        "reason": reason
    })

# --------------------------------------------------
# SAVE MANIFEST
# --------------------------------------------------

manifest_path = OUTPUT_DIR / "training_manifest.csv"

with open(
    manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fields = [
        "tile_id",
        "mining_type",
        "activity_type",
        "coverage_percent",
        "status",
        "reason"
    ]

    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(manifest)

# --------------------------------------------------
# PRINT SUMMARY
# --------------------------------------------------

print("\nMINING TRAINING DATASET PREPARATION")
print("-----------------------------------")

print("Original metadata samples:", len(metadata))
print("Included samples:", included)
print("Excluded samples:", excluded)

print("\nIncluded category distribution:")

for mining_type, activity_type in [
    ("Placer", "Artisanal"),
    ("Placer", "Industrial"),
    ("Surface", "Artisanal"),
    ("Surface", "Industrial")
]:

    count = sum(
        1 for row in manifest
        if row["status"] == "INCLUDED"
        and row["mining_type"] == mining_type
        and row["activity_type"] == activity_type
    )

    print(f"{mining_type} - {activity_type}: {count}")

print("\nTraining dataset:", OUTPUT_DIR)
print("Manifest:", manifest_path)

print("\nDataset preparation completed.")
