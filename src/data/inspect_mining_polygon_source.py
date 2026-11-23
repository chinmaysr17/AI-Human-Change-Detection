
import sqlite3
from pathlib import Path
from shapely import wkb

GPKG = Path("dataset/Mining/extracted/mining_area_data.gpkg")

def decode_geometry(blob):
    flags = blob[3]
    code = (flags >> 1) & 0b111
    sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    return wkb.loads(blob[8 + sizes[code]:])

connection = sqlite3.connect(GPKG)

print("\nGEOPACKAGE CRS INFORMATION")
print("-" * 45)

for row in connection.execute(
    "SELECT table_name, column_name, geometry_type_name, srs_id "
    "FROM gpkg_geometry_columns"
):
    print(row)

print("\nGPKG CONTENTS")
print("-" * 45)

for row in connection.execute(
    "SELECT table_name, data_type, srs_id "
    "FROM gpkg_contents"
):
    print(row)

print("\nTILE 2120 POLYGON INFORMATION")
print("-" * 45)

rows = connection.execute(
    "SELECT geom FROM preferred_polygons WHERE tile_id = ?",
    (2120,)
).fetchall()

print("Matching polygon records:", len(rows))

for index, row in enumerate(rows, start=1):
    polygon = decode_geometry(row[0])

    print(f"\nPolygon {index}")
    print("Geometry type:", polygon.geom_type)
    print("Valid:", polygon.is_valid)
    print("Bounds:", polygon.bounds)
    print("Area:", polygon.area)
    print("Empty:", polygon.is_empty)

    header_srs_id = int.from_bytes(row[0][4:8], "little", signed=True)
    print("Geometry header SRS ID:", header_srs_id)

connection.close()
