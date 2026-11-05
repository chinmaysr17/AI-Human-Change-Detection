
import sqlite3

path = "dataset/Mining/extracted/mining_area_data.gpkg"

conn = sqlite3.connect(path)
cursor = conn.cursor()

cursor.execute("""
    SELECT
        tile_id,
        s2_tile_id,
        source_dataset,
        preferred_dataset,
        minetype1,
        minetype2,
        timestamp,
        split
    FROM tiles
    LIMIT 10
""")

print("SAMPLE SENTINEL-2 TILE RECORDS")
print("=" * 70)

for row in cursor.fetchall():
    for name, value in zip(
        ["Tile ID", "Sentinel-2 ID", "Source",
         "Preferred", "Mining Type 1", "Mining Type 2",
         "Timestamp", "Split"],
        row
    ):
        print(f"{name}: {value}")
    print("-" * 40)

cursor.execute("""
    SELECT COUNT(DISTINCT s2_tile_id)
    FROM tiles
""")

print("Unique Sentinel-2 IDs:", cursor.fetchone()[0])

conn.close()
