import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import ndimage

BEFORE = "data/sentinel2/processed/masked/sentinel2_20260927_shivamogga_10km_masked.tif"
AFTER = "data/sentinel2/processed/masked/sentinel2_20261027_shivamogga_10km_masked.tif"

DIFFERENCE = "data/sentinel2/processed/change_analysis/sentinel2_band_differences.tif"

OUTPUT_DIR = "data/sentinel2/processed/change_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with rasterio.open(BEFORE) as src:
    before = src.read()
    profile = src.profile.copy()
    pixel_area = abs(src.transform.a * src.transform.e)

with rasterio.open(AFTER) as src:
    after = src.read()

with rasterio.open(DIFFERENCE) as src:
    difference = src.read().astype(np.float32)

# Correct common valid-pixel mask
valid = np.all(before > 0, axis=0) & np.all(after > 0, axis=0)

# Mean absolute spectral difference
mean_difference = difference.mean(axis=0)

values = mean_difference[valid]

# Exploratory 98th percentile threshold
threshold = np.percentile(values, 98)

candidate = (mean_difference >= threshold) & valid

# Remove small isolated regions
structure = np.ones((3, 3), dtype=bool)
labels, count = ndimage.label(candidate, structure=structure)

cleaned = np.zeros(candidate.shape, dtype=np.uint8)

for region_id in range(1, count + 1):
    region = labels == region_id

    if region.sum() >= 4:
        cleaned[region] = 1

# Correct area calculation
candidate_pixels = int(cleaned.sum())
area_m2 = candidate_pixels * pixel_area
area_hectares = area_m2 / 10000

# Save binary map
output_profile = profile.copy()
output_profile.update(
    count=1,
    dtype="uint8",
    nodata=255,
    compress="deflate"
)

binary_path = os.path.join(
    OUTPUT_DIR, "candidate_change_map.tif"
)

result = cleaned.copy()
result[~valid] = 255

with rasterio.open(binary_path, "w", **output_profile) as dst:
    dst.write(result, 1)

# Generate preview
plt.figure(figsize=(10, 8))

plt.imshow(
    np.ma.masked_where(~valid, cleaned),
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    f"Exploratory Candidate Change Map\n"
    f"98th Percentile Threshold: {threshold:.2f}"
)

plt.axis("off")
plt.tight_layout()

preview_path = os.path.join(
    OUTPUT_DIR, "candidate_change_preview.png"
)

plt.savefig(preview_path, dpi=200)
plt.close()

print("Candidate change mapping completed.")
print("Common valid pixels:", int(valid.sum()))
print("Threshold:", round(float(threshold), 2))
print("Candidate pixels before filtering:", int(candidate.sum()))
print("Candidate pixels after filtering:", candidate_pixels)
print("Candidate area (hectares):", round(area_hectares, 2))
print(
    "Candidate area percentage:",
    round(candidate_pixels / valid.sum() * 100, 2)
)
print("Binary map:", binary_path)
print("Preview:", preview_path)