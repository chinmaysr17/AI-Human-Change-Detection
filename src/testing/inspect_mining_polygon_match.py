
import geopandas as gpd
import sqlite3

DB_PATH = "dataset/Mining/extracted/mining_area_data.gpkg"

PRODUCT_ID = (
    "S2A_MSIL2A_20191017T004711_R102_"
    "T54LWM_20201004T031000"
)

# Read tile metadata
conn = sqlite3.connect(DB_PATH)

tiles = gpd.read_file(
    DB_PATH,
    layer="tiles"
)

match = tiles[tiles["s2_tile_id"] == PRODUCT_ID]

print("MATCHING TILE RECORDS:", len(match))

if match.empty:
    print("No matching tile metadata found.")
    conn.close()
    raise SystemExit

tile_ids = match["tile_id"].tolist()

print("TILE IDs:", tile_ids)
print("TILE CRS:", match.crs)

# Read preferred mining polygons
polygons = gpd.read_file(
    DB_PATH,
    layer="preferred_polygons"
)

selected = polygons[
    polygons["tile_id"].isin(tile_ids)
]

print("\nMINING POLYGON INFORMATION")
print("Polygon count:", len(selected))
print("Polygon CRS:", selected.crs)
print("Geometry types:", selected.geometry.geom_type.value_counts().to_dict())

if not selected.empty:
    print("\nFIRST POLYGON BOUNDS:")
    print(selected.geometry.iloc[0].bounds)

    print("\nFIRST POLYGON AREA:")
    print(selected.geometry.iloc[0].area)

conn.close()
