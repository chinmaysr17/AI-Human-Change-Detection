
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "dataset/Mining/extracted/mining_area_data.gpkg"

connection = sqlite3.connect(DB)

print("MINING TYPE DISTRIBUTION")
print("-" * 45)

for row in connection.execute("""
    SELECT minetype1, COUNT(*)
    FROM tiles
    WHERE split = 'train'
      AND source_dataset != 'rejected'
    GROUP BY minetype1
    ORDER BY COUNT(*) DESC
"""):
    print(row)

print("\nINDUSTRY TYPE DISTRIBUTION")
print("-" * 45)

for row in connection.execute("""
    SELECT minetype2, COUNT(*)
    FROM tiles
    WHERE split = 'train'
      AND source_dataset != 'rejected'
    GROUP BY minetype2
    ORDER BY COUNT(*) DESC
"""):
    print(row)

connection.close()
