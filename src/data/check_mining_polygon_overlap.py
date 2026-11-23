
import sqlite3
import csv
from pathlib import Path

import planetary_computer
import rasterio
import requests

from shapely import wkb
from shapely.geometry import shape, mapping, box
from rasterio.warp import transform_geom
from rasterio.windows import Window

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

GPKG = Path("dataset/Mining/extracted/mining_area_data.gpkg")
CSV_FILE = Path("dataset/Mining/mining_training_30.csv")

STAC_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1/"
    "collections/sentinel-2-l2a/items/"
)

TILE_ID = 783
SIZE = 256

# --------------------------------------------------
# 1. LOAD MINING POLYGON
# --------------------------------------------------

connection = sqlite3.connect(GPKG)

row = connection.execute(
    "SELECT geom FROM preferred_polygons WHERE tile_id = ?",
    (TILE_ID,)
).fetchone()

connection.close()

if row is None:
    raise RuntimeError("Mining polygon not found.")

blob = row[0]

flags = blob[3]
code = (flags >> 1) & 0b111
sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}

polygon = wkb.loads(blob[8 + sizes[code]:])

# --------------------------------------------------
# 2. LOAD SENTINEL-2 PRODUCT ID
# --------------------------------------------------

with open(CSV_FILE, newline="", encoding="utf-8") as file:
    records = list(csv.DictReader(file))

record = next(
    (r for r in records if str(r["tile_id"]) == str(TILE_ID)),
    None
)

if record is None:
    raise RuntimeError("Tile not found in training CSV.")

product_id = record["s2_tile_id"]

print("Tile ID:", TILE_ID)
print("Sentinel-2 product:", product_id)

# --------------------------------------------------
# 3. RETRIEVE SENTINEL-2 PRODUCT
# --------------------------------------------------

response = requests.get(
    STAC_URL + product_id,
    timeout=60
)
response.raise_for_status()

item = planetary_computer.sign(response.json())

# --------------------------------------------------
# 4. TRANSFORM POLYGON AND CREATE CHIP WINDOW
# --------------------------------------------------

with rasterio.open(item["assets"]["B04"]["href"]) as src:

    projected = shape(
        transform_geom(
            "EPSG:4326",
            src.crs,
            mapping(polygon)
        )
    )

    x, y = projected.centroid.coords[0]
    row, col = src.index(x, y)

    window = Window(
        col - SIZE // 2,
        row - SIZE // 2,
        SIZE,
        SIZE
    )

    transform = src.window_transform(window)

    chip_bounds = box(
        *rasterio.windows.bounds(window, src.transform)
    )

    intersection = projected.intersection(chip_bounds)

    # --------------------------------------------------
    # 5. CALCULATE OVERLAP
    # --------------------------------------------------

    polygon_area = projected.area
    chip_area = chip_bounds.area
    intersection_area = intersection.area

    polygon_coverage = (
        intersection_area / chip_area * 100
        if chip_area else 0
    )

    chip_coverage = (
        intersection_area / polygon_area * 100
        if polygon_area else 0
    )

    print("\nPOLYGON AND CHIP COMPARISON")
    print("---------------------------")
    print("Raster CRS:", src.crs)
    print("Raster resolution:", src.res)
    print("Polygon bounds:", projected.bounds)
    print("Chip bounds:", chip_bounds.bounds)
    print("Polygon area:", polygon_area)
    print("Chip area:", chip_area)
    print("Intersection area:", intersection_area)

    print(
        "Polygon coverage of chip:",
        round(polygon_coverage, 2), "%"
    )

    print(
        "Chip coverage of polygon:",
        round(chip_coverage, 2), "%"
    )

print("\nOverlap analysis completed.")
