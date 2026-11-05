
import sqlite3
from pathlib import Path
from shapely import wkb

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "dataset/Mining/extracted/mining_area_data.gpkg"

connection = sqlite3.connect(DB)

row = connection.execute(
    "SELECT geom FROM preferred_polygons WHERE tile_id=783 LIMIT 1"
).fetchone()

connection.close()

if row is None:
    raise RuntimeError("Polygon not found.")

blob = row[0]
code = (blob[3] >> 1) & 7
sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}

# Correct GeoPackage geometry decoding.
polygon = wkb.loads(blob[8 + sizes[code]:])

print("MINING COMPONENT ANALYSIS")
print("-" * 60)

components = sorted(
    polygon.geoms,
    key=lambda g: g.area,
    reverse=True
)

for index, component in enumerate(components, start=1):
    point = component.representative_point()

    print(f"\nComponent {index}")
    print(f"Area: {component.area:.8f}")
    print(f"Center: ({point.x:.6f}, {point.y:.6f})")
    print(f"Bounds: {component.bounds}")

print("\nTotal components:", len(components))
