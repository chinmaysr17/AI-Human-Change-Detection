import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

BEFORE = (
    "data/sentinel2/processed/masked/"
    "sentinel2_20260927_shivamogga_10km_masked.tif"
)

AFTER = (
    "data/sentinel2/processed/masked/"
    "sentinel2_20261027_shivamogga_10km_masked.tif"
)

CANDIDATE = (
    "data/sentinel2/processed/change_analysis/"
    "candidate_change_map.tif"
)

OUTPUT_DIR = "data/sentinel2/processed/change_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read October satellite image
with rasterio.open(AFTER) as src:
    after = src.read()
    profile = src.profile.copy()

# Read September image to verify common valid pixels
with rasterio.open(BEFORE) as src:
    before = src.read()

# Read candidate change map
with rasterio.open(CANDIDATE) as src:
    candidate = src.read(1)

# Calculate common valid area
valid = (
    np.all(before > 0, axis=0)
    & np.all(after > 0, axis=0)
)

# Candidate pixels, excluding invalid locations
change_mask = (candidate == 1) & valid

# Natural colour: B04 (Red), B03 (Green), B02 (Blue)
rgb = np.stack(
    [after[2], after[1], after[0]],
    axis=-1
).astype(np.float32)

# Apply percentile contrast stretching
for band in range(3):
    values = rgb[:, :, band][valid]

    low, high = np.percentile(values, [2, 98])

    rgb[:, :, band] = np.clip(
        (rgb[:, :, band] - low) / (high - low),
        0,
        1
    )

# Create overlay
overlay = np.zeros(
    (change_mask.shape[0], change_mask.shape[1], 4)
)

# Red colour for candidate changes
overlay[change_mask] = [1, 0, 0, 0.65]

plt.figure(figsize=(12, 10))

plt.imshow(rgb)
plt.imshow(overlay)

plt.title(
    "Sentinel-2 Candidate Change Overlay\n"
    "27 October 2026"
)

plt.axis("off")

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "candidate_change_overlay.png"
)

plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print("Candidate change overlay generated.")
print("Candidate pixels:", int(change_mask.sum()))
print("Candidate area (hectares):", round(change_mask.sum() * 0.01, 2))
print("Overlay saved:", output_path)