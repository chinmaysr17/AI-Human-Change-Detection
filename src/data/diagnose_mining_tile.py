
import sqlite3
from pathlib import Path

from shapely import wkb

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "dataset/Mining/extracted/mining_area_data.gpkg"

connection = sqlite3.connect(DB)

for tile_id in [783]:
    row = connection.execute(
        "SELECT geom FROM preferred_polygons WHERE tile_id=? LIMIT 1",
        (tile_id,)
    ).fetchone()

    if row is None:
        print("Polygon not found:", tile_id)
        continue

    blob = row[0]

    # Decode GeoPackage geometry header.
    code = (blob[3] >> 1) & 7
    sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}

    offset = 8 + sizes[code]
    polygon = wkb.loads(blob[offset:])

    print("\nTILE:", tile_id)
    print("Geometry:", polygon.geom_type)
    print("Bounds:", polygon.bounds)
    print("Area:", polygon.area)
    print("Centroid:", polygon.centroid.coords[0])
    print("Representative point:",
          polygon.representative_point().coords[0])

    if polygon.geom_type == "MultiPolygon":
        parts = sorted(
            polygon.geoms,
            key=lambda g: g.area,
            reverse=True
        )

        print("Number of components:", len(parts))

        for i, part in enumerate(parts[:5]):
            print(
                f"\nComponent {i + 1}",
                "\nArea:", part.area,
                "\nCentroid:", part.centroid.coords[0],
                "\nRepresentative:",
                part.representative_point().coords[0]
            )

connection.close()
