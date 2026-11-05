
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "dataset/Mining/extracted/mining_area_data.gpkg"

connection = sqlite3.connect(DB)

query = """
SELECT minetype1, minetype2, COUNT(*)
FROM tiles
WHERE split = 'train'
AND source_dataset != 'rejected'
GROUP BY minetype1, minetype2
ORDER BY minetype1, minetype2
"""

print("MINING TYPE × ACTIVITY TYPE")
print("-" * 50)

for mining_type, activity_type, count in connection.execute(query):
    print(f"{mining_type:12} | {activity_type:12} | {count:4}")

connection.close()
