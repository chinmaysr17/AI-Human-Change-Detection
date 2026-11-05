
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "dataset/Mining/extracted/mining_area_data.gpkg"

connection = sqlite3.connect(DB)

query = """
SELECT tile_id, s2_tile_id, minetype1, minetype2
FROM tiles
WHERE split = 'train'
AND source_dataset != 'rejected'
LIMIT 5
"""

rows = connection.execute(query).fetchall()
connection.close()

print("FIVE MINING TRAINING SAMPLES\n")

for row in rows:
    print("Tile ID:", row[0])
    print("Product:", row[1])
    print("Mining type:", row[2])
    print("Activity:", row[3])
    print("-" * 60)
