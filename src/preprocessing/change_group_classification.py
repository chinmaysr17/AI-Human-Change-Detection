import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

BASE = "data/sentinel2/processed"
CHANGE_DIR = os.path.join(BASE, "change_analysis")
MASK_DIR = os.path.join(BASE, "masked")

BEFORE = os.path.join(
    MASK_DIR, "sentinel2_20260927_shivamogga_10km_masked.tif"
)

AFTER = os.path.join(
    MASK_DIR, "sentinel2_20261027_shivamogga_10km_masked.tif"
)

NDVI_FILE = os.path.join(CHANGE_DIR, "ndvi_difference.tif")
CANDIDATE_FILE = os.path.join(
    CHANGE_DIR, "candidate_change_map.tif"
)

with rasterio.open(BEFORE) as src:
    before = src.read()
    profile = src.profile.copy()

with rasterio.open(AFTER) as src:
    after = src.read()

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

# Use the existing cleaned candidate map
candidate_mask = (candidate == 1) & valid

# Exploratory NDVI threshold
threshold = 0.0815

# Group 1: High vegetation-index variation
vegetation_group = (
    candidate_mask & (np.abs(ndvi) >= threshold)
)

# Group 2: Other spectral changes
other_group = (
    candidate_mask & (np.abs(ndvi) < threshold)
)

# Create classification raster
# 0 = No candidate change
# 1 = Vegetation-associated spectral change
# 2 = Other spectral change
# 255 = Invalid pixel

classified = np.zeros(valid.shape, dtype=np.uint8)
classified[vegetation_group] = 1
classified[other_group] = 2
classified[~valid] = 255

# Area calculation
pixel_area_m2 = abs(profile["transform"].a * profile["transform"].e)
pixel_area_ha = pixel_area_m2 / 10000

veg_pixels = int(vegetation_group.sum())
other_pixels = int(other_group.sum())
total_pixels = int(candidate_mask.sum())

# Save classification raster
output_profile = profile.copy()
output_profile.update(
    count=1,
    dtype="uint8",
    nodata=255,
    compress="deflate"
)

output_path = os.path.join(
    CHANGE_DIR, "change_group_classification.tif"
)

with rasterio.open(output_path, "w", **output_profile) as dst:
    dst.write(classified, 1)

# Create visualization
display = np.ma.masked_where(~valid, classified)

plt.figure(figsize=(10, 8))
plt.imshow(display, cmap="viridis", vmin=0, vmax=2)

plt.title("Exploratory Change Group Classification")
plt.colorbar(
    ticks=[0, 1, 2],
    label="0: No candidate | 1: Vegetation-associated | 2: Other"
)

plt.axis("off")
plt.tight_layout()

preview_path = os.path.join(
    CHANGE_DIR, "change_group_classification.png"
)

plt.savefig(preview_path, dpi=200)
plt.close()

print("Change group classification completed.")
print("NDVI threshold:", threshold)
print("Total candidate pixels:", total_pixels)

print("\nGROUP 1: Vegetation-associated")
print("Pixels:", veg_pixels)
print("Area (hectares):", round(veg_pixels * pixel_area_ha, 2))

print("\nGROUP 2: Other spectral changes")
print("Pixels:", other_pixels)
print("Area (hectares):", round(other_pixels * pixel_area_ha, 2))

print("\nTotal candidate area (hectares):",
      round(total_pixels * pixel_area_ha, 2))

print("Classification raster:", output_path)
print("Preview:", preview_path)