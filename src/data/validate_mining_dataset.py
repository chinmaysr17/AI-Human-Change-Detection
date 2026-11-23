
from pathlib import Path
import csv
import numpy as np

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("dataset/Mining/training_chips_30")
OUTPUT_DIR = Path("dataset/Mining")
REPORT_PATH = OUTPUT_DIR / "mining_dataset_quality_report.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_files = sorted(DATA_DIR.glob("mining_*.npy"))

results = []

# --------------------------------------------------
# DATASET VALIDATION
# --------------------------------------------------

for image_path in image_files:

    tile_id = image_path.stem.replace("mining_", "")
    mask_path = DATA_DIR / f"mask_{tile_id}.npy"

    issues = []

    # Check whether corresponding mask exists
    if not mask_path.exists():
        results.append({
            "tile_id": tile_id,
            "image_shape": "",
            "mask_shape": "",
            "image_dtype": "",
            "mask_dtype": "",
            "coverage_percent": "",
            "boundary_touch_percent": "",
            "status": "REJECT",
            "issues": "Missing mask"
        })
        continue

    try:
        image = np.load(image_path)
        mask = np.load(mask_path)

        # ------------------------------------------
        # 1. Validate dimensions
        # ------------------------------------------

        if image.shape != (256, 256, 4):
            issues.append(f"Unexpected image shape: {image.shape}")

        if mask.shape != (256, 256):
            issues.append(f"Unexpected mask shape: {mask.shape}")

        # ------------------------------------------
        # 2. Validate image values
        # ------------------------------------------

        if not np.issubdtype(image.dtype, np.number):
            issues.append("Image is not numeric")

        elif not np.isfinite(image).all():
            issues.append("Image contains NaN or infinite values")

        # ------------------------------------------
        # 3. Validate mask values
        # ------------------------------------------

        unique_values = np.unique(mask)

        if not np.isin(unique_values, [0, 1, 255]).all():
            issues.append(f"Unexpected mask values: {unique_values}")

        # ------------------------------------------
        # 4. Calculate mask coverage
        # ------------------------------------------

        positive = mask > 0

        coverage = float(np.mean(positive) * 100)

        # ------------------------------------------
        # 5. Calculate boundary contact
        # ------------------------------------------

        boundary = np.zeros(mask.shape, dtype=bool)

        boundary[0, :] = True
        boundary[-1, :] = True
        boundary[:, 0] = True
        boundary[:, -1] = True

        boundary_pixels = int(np.sum(positive & boundary))

        boundary_touch = (
            boundary_pixels / max(int(np.sum(positive)), 1)
        ) * 100

        # ------------------------------------------
        # 6. Flag suspicious masks
        # ------------------------------------------

        if coverage == 0:
            issues.append("Empty mask")

        elif coverage < 1:
            issues.append("Very low mask coverage")

        if coverage > 95:
            issues.append("Very high mask coverage")

        if boundary_touch > 5:
            issues.append(
                "High boundary contact - visual inspection required"
            )

        # ------------------------------------------
        # 7. Determine sample status
        # ------------------------------------------

        status = "REVIEW" if issues else "PASS"

        results.append({
            "tile_id": tile_id,
            "image_shape": str(image.shape),
            "mask_shape": str(mask.shape),
            "image_dtype": str(image.dtype),
            "mask_dtype": str(mask.dtype),
            "coverage_percent": round(coverage, 2),
            "boundary_touch_percent": round(boundary_touch, 2),
            "status": status,
            "issues": "; ".join(issues) if issues else "None"
        })

    except Exception as error:

        results.append({
            "tile_id": tile_id,
            "image_shape": "",
            "mask_shape": "",
            "image_dtype": "",
            "mask_dtype": "",
            "coverage_percent": "",
            "boundary_touch_percent": "",
            "status": "REJECT",
            "issues": str(error)
        })

# --------------------------------------------------
# SAVE QUALITY REPORT
# --------------------------------------------------

fieldnames = [
    "tile_id",
    "image_shape",
    "mask_shape",
    "image_dtype",
    "mask_dtype",
    "coverage_percent",
    "boundary_touch_percent",
    "status",
    "issues"
]

with open(
    REPORT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

# --------------------------------------------------
# PRINT VALIDATION SUMMARY
# --------------------------------------------------

passed = sum(r["status"] == "PASS" for r in results)
review = sum(r["status"] == "REVIEW" for r in results)
rejected = sum(r["status"] == "REJECT" for r in results)

print("\nMINING DATASET QUALITY VALIDATION")
print("---------------------------------")

print(f"Image files found : {len(image_files)}")
print(f"Passed            : {passed}")
print(f"Needs review      : {review}")
print(f"Rejected          : {rejected}")

print(f"\nReport saved to: {REPORT_PATH}")

print("\nSamples requiring attention:")

for row in results:

    if row["status"] != "PASS":

        print(
            f"Tile {row['tile_id']}: "
            f"{row['status']} - {row['issues']}"
        )

print("\nValidation completed successfully.")
