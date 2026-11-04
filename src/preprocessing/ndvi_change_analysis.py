import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

BEFORE = "data/sentinel2/processed/masked/sentinel2_20260927_shivamogga_10km_masked.tif"
AFTER = "data/sentinel2/processed/masked/sentinel2_20261027_shivamogga_10km_masked.tif"
CANDIDATE = "data/sentinel2/processed/change_analysis/candidate_change_map.tif"

OUTPUT_DIR = "data/sentinel2/processed/change_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with rasterio.open(BEFORE) as src:
    before = src.read().astype(np.float32)
    profile = src.profile.copy()

with rasterio.open(AFTER) as src:
    after = src.read().astype(np.float32)

with rasterio.open(CANDIDATE) as src:
    candidate = src.read(1)

# Band order: Blue, Green, Red, NIR
red_before, nir_before = before[2], before[3]
red_after, nir_after = after[2], after[3]

# Common valid pixels
valid = (
    np.all(before > 0, axis=0)
    & np.all(after > 0, axis=0)
)

# Calculate NDVI
ndvi_before = np.zeros(valid.shape, dtype=np.float32)
ndvi_after = np.zeros(valid.shape, dtype=np.float32)

den_before = nir_before + red_before
den_after = nir_after + red_after

ndvi_before[valid] = (
    (nir_before[valid] - red_before[valid])
    / np.maximum(den_before[valid], 1)
)

ndvi_after[valid] = (
    (nir_after[valid] - red_after[valid])
    / np.maximum(den_after[valid], 1)
)

# NDVI difference
ndvi_difference = ndvi_after - ndvi_before
absolute_difference = np.abs(ndvi_difference)

values = absolute_difference[valid]

# Exploratory statistics
threshold = np.percentile(values, 90)

candidate_mask = (candidate == 1) & valid
high_vegetation_change = (
    (absolute_difference >= threshold) & valid
)

overlap = candidate_mask & high_vegetation_change

# Save NDVI difference raster
output_profile = profile.copy()
output_profile.update(
    count=1,
    dtype="float32",
    nodata=-9999,
    compress="deflate"
)

ndvi_path = os.path.join(
    OUTPUT_DIR, "ndvi_difference.tif"
)

output = ndvi_difference.copy()
output[~valid] = -9999

with rasterio.open(ndvi_path, "w", **output_profile) as dst:
    dst.write(output, 1)

# Generate NDVI change preview
plt.figure(figsize=(10, 8))

plt.imshow(
    np.ma.masked_where(~valid, ndvi_difference),
    cmap="RdYlGn",
    vmin=-0.5,
    vmax=0.5
)

plt.colorbar(label="NDVI Change (October - September)")
plt.title("Sentinel-2 NDVI Change\n27 September – 27 October 2026")
plt.axis("off")
plt.tight_layout()

preview_path = os.path.join(
    OUTPUT_DIR, "ndvi_change_preview.png"
)

plt.savefig(preview_path, dpi=200)
plt.close()

print("NDVI change analysis completed.")
print("Mean NDVI change:", round(float(ndvi_difference[valid].mean()), 4))
print("Median absolute NDVI change:", round(float(np.median(values)), 4))
print("90th percentile absolute NDVI change:", round(float(threshold), 4))
print("Candidate pixels:", int(candidate_mask.sum()))
print("High vegetation-change pixels:", int(high_vegetation_change.sum()))
print("Overlap pixels:", int(overlap.sum()))
print("NDVI raster:", ndvi_path)
print("Preview:", preview_path)