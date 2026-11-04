import os
import csv
import numpy as np
import rasterio
from scipy import ndimage

BASE = "data/sentinel2/processed"
MASK_DIR = os.path.join(BASE, "masked")
CHANGE_DIR = os.path.join(BASE, "change_analysis")

BEFORE = os.path.join(
    MASK_DIR, "sentinel2_20260927_shivamogga_10km_masked.tif"
)

AFTER = os.path.join(
    MASK_DIR, "sentinel2_20261027_shivamogga_10km_masked.tif"
)

DIFFERENCE = os.path.join(
    CHANGE_DIR, "sentinel2_band_differences.tif"
)

NDVI_FILE = os.path.join(
    CHANGE_DIR, "ndvi_difference.tif"
)

CANDIDATE_FILE = os.path.join(
    CHANGE_DIR, "candidate_change_map.tif"
)

OUTPUT_FILE = os.path.join(
    CHANGE_DIR, "region_validation_report.csv"
)

# Read satellite images
with rasterio.open(BEFORE) as src:
    before = src.read()
    transform = src.transform
    pixel_area = abs(transform.a * transform.e)

with rasterio.open(AFTER) as src:
    after = src.read()

with rasterio.open(DIFFERENCE) as src:
    difference = src.read().astype(np.float32)

with rasterio.open(NDVI_FILE) as src:
    ndvi = src.read(1)

with rasterio.open(CANDIDATE_FILE) as src:
    candidate = src.read(1)

# Common valid pixels
valid = (
    np.all(before > 0, axis=0)
    & np.all(after > 0, axis=0)
    & (ndvi != -9999)
)

# Candidate regions
candidate_mask = (candidate == 1) & valid

labels, region_count = ndimage.label(
    candidate_mask,
    structure=np.ones((3, 3), dtype=bool)
)

# Mean absolute spectral difference
mean_spectral = np.mean(difference, axis=0)

# Prepare CSV report
headers = [
    "Region_ID",
    "Pixel_Count",
    "Area_Hectares",
    "Mean_Spectral_Difference",
    "Mean_NDVI_Change",
    "Mean_Absolute_NDVI_Change",
    "Centroid_X",
    "Centroid_Y"
]

rows = []

for region_id in range(1, region_count + 1):

    region = labels == region_id
    pixel_count = int(region.sum())

    # Region centre in pixel coordinates
    row_indices, col_indices = np.where(region)

    centre_row = row_indices.mean()
    centre_col = col_indices.mean()

    # Convert pixel centre to projected coordinates
    centre_x, centre_y = transform * (
        centre_col + 0.5,
        centre_row + 0.5
    )

    area_hectares = pixel_count * pixel_area / 10000

    rows.append([
        region_id,
        pixel_count,
        round(area_hectares, 4),
        round(float(mean_spectral[region].mean()), 3),
        round(float(ndvi[region].mean()), 5),
        round(float(np.abs(ndvi[region]).mean()), 5),
        round(centre_x, 2),
        round(centre_y, 2)
    ])

# Sort largest regions first
rows.sort(key=lambda x: x[1], reverse=True)

# Write CSV
with open(OUTPUT_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(rows)

# Display summary
print("Region-level validation report generated.")
print("Total detected regions:", region_count)
print("Total candidate pixels:", int(candidate_mask.sum()))
print(
    "Total candidate area (hectares):",
    round(candidate_mask.sum() * pixel_area / 10000, 2)
)

print("\nTOP 10 LARGEST REGIONS")

for row in rows[:10]:
    print(
        f"Region {row[0]} | "
        f"Pixels: {row[1]} | "
        f"Area: {row[2]} ha | "
        f"Spectral difference: {row[3]} | "
        f"Mean NDVI change: {row[4]}"
    )

print("\nCSV report:", OUTPUT_FILE)