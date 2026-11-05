
import requests
import rasterio
import planetary_computer
from pystac import Item

DB_PATH = "dataset/Mining/extracted/mining_area_data.gpkg"

import sqlite3

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT s2_tile_id
    FROM tiles
    WHERE s2_tile_id IS NOT NULL
    LIMIT 1
""")

product_id = cursor.fetchone()[0]
conn.close()

url = (
    "https://planetarycomputer.microsoft.com/api/stac/v1/"
    "collections/sentinel-2-l2a/items/" + product_id
)

response = requests.get(url, timeout=60)
response.raise_for_status()

item = Item.from_dict(response.json())
item = planetary_computer.sign(item)

print("PRODUCT:", product_id)

for band in ["B02", "B03", "B04", "B08", "SCL"]:
    print("\nChecking band:", band)

    asset = item.assets[band]

    with rasterio.open(asset.href) as src:
        print("CRS:", src.crs)
        print("Dimensions:", src.width, "x", src.height)
        print("Resolution:", src.res)
        print("Data type:", src.dtypes[0])
        print("Bounds:", src.bounds)

print("\nSUCCESS: Satellite raster assets are readable.")
