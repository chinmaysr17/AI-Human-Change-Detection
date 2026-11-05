
import sqlite3
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely import wkb
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[2]

GPKG = ROOT / "dataset/Mining/extracted/mining_area_data.gpkg"
CHIP = ROOT / "data/sentinel2/mining/mining_2805_B04.tif"
OUTPUT = ROOT / "dataset/Mining/chips"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Read the preferred mining polygon.
connection = sqlite3.connect(GPKG)

row = connection.execute(
    "SELECT geom FROM preferred_polygons WHERE tile_id = 2805 LIMIT 1"
).fetchone()

connection.close()

if row is None:
    raise RuntimeError("Mining polygon not found.")

blob = row[0]
envelope_code = (blob[3] >> 1) & 0b111
envelope_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}

polygon = wkb.loads(blob[8 + envelope_sizes[envelope_code]:])

# Match polygon to the downloaded image chip.
with rasterio.open(CHIP) as src:
    projected_polygon = shape(
        transform_geom(
            "EPSG:4326",
            src.crs,
            mapping(polygon)
        )
    )

    mask = rasterize(
        [(mapping(projected_polygon), 1)],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8"
    )

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw"
    )

    mask_path = OUTPUT / "mining_mask_2805.tif"

    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)

np.save(OUTPUT / "mining_mask_2805.npy", mask)

print("MASK CREATED SUCCESSFULLY")
print("Mask shape:", mask.shape)
print("Mining pixels:", int(mask.sum()))
print("Background pixels:", int((mask == 0).sum()))
print("Mining coverage:", round(float(mask.mean() * 100), 2), "%")
print("Saved:", mask_path)
