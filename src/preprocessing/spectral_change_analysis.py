import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

BEFORE = "data/sentinel2/processed/masked/sentinel2_20260927_shivamogga_10km_masked.tif"
AFTER = "data/sentinel2/processed/masked/sentinel2_20261027_shivamogga_10km_masked.tif"

OUTPUT_DIR = "data/sentinel2/processed/change_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with rasterio.open(BEFORE) as src:
    before = src.read().astype(np.float32)
    profile = src.profile.copy()

with rasterio.open(AFTER) as src:
    after = src.read().astype(np.float32)

# Common valid pixels
valid = np.all(before > 0, axis=0) & np.all(after > 0, axis=0)

# Absolute spectral difference
difference = np.abs(after - before)

# Mean difference across four spectral bands
mean_difference = np.mean(difference, axis=0)

# Ignore invalid pixels
mean_difference[~valid] = 0
difference[:, ~valid] = 0

# Save four-band difference image
diff_profile = profile.copy()
diff_profile.update(
    count=4,
    dtype="float32",
    nodata=0,
    compress="deflate"
)

diff_path = os.path.join(
    OUTPUT_DIR, "sentinel2_band_differences.tif"
)

with rasterio.open(diff_path, "w", **diff_profile) as dst:
    dst.write(difference)

# Save mean difference image
mean_profile = profile.copy()
mean_profile.update(
    count=1,
    dtype="float32",
    nodata=0,
    compress="deflate"
)

mean_path = os.path.join(
    OUTPUT_DIR, "sentinel2_mean_difference.tif"
)

with rasterio.open(mean_path, "w", **mean_profile) as dst:
    dst.write(mean_difference, 1)

# Generate visualization
values = mean_difference[valid]

vmin, vmax = np.percentile(values, [2, 98])

plt.figure(figsize=(10, 8))
plt.imshow(
    np.ma.masked_where(~valid, mean_difference),
    cmap="inferno",
    vmin=vmin,
    vmax=vmax
)

plt.colorbar(label="Mean Absolute Spectral Difference (DN)")
plt.title("Sentinel-2 Spectral Difference\n27 September – 27 October 2026")
plt.axis("off")
plt.tight_layout()

preview_path = os.path.join(
    OUTPUT_DIR, "sentinel2_difference_preview.png"
)

plt.savefig(preview_path, dpi=200)
plt.close()

print("Spectral difference analysis completed.")
print("Common valid pixels:", int(valid.sum()))
print("Mean difference:", round(float(values.mean()), 2))
print("Median difference:", round(float(np.median(values)), 2))
print("95th percentile:", round(float(np.percentile(values, 95)), 2))
print("Difference image:", diff_path)
print("Mean difference image:", mean_path)
print("Preview:", preview_path)