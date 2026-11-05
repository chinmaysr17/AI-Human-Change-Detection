
import csv
import random
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DB = ROOT / "dataset/Mining/extracted/mining_area_data.gpkg"
OUTPUT = ROOT / "dataset/Mining/mining_training_30.csv"

SELECTION = {
    ("Placer", "Artisanal"): 8,
    ("Placer", "Industrial"): 5,
    ("Surface", "Artisanal"): 7,
    ("Surface", "Industrial"): 10,
}

random.seed(42)

connection = sqlite3.connect(DB)

selected = []

for (mining_type, activity), count in SELECTION.items():
    rows = connection.execute("""
        SELECT tile_id, s2_tile_id, minetype1, minetype2
        FROM tiles
        WHERE split = 'train'
          AND source_dataset != 'rejected'
          AND minetype1 = ?
          AND minetype2 = ?
    """, (mining_type, activity)).fetchall()

    if len(rows) < count:
        raise RuntimeError(
            f"Not enough samples for {mining_type}-{activity}"
        )

    chosen = random.sample(rows, count)
    selected.extend(chosen)

connection.close()

with open(OUTPUT, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "tile_id",
        "s2_tile_id",
        "mining_type",
        "activity_type"
    ])
    writer.writerows(selected)

print("SELECTED MINING TRAINING SAMPLES")
print("-" * 50)

for category, count in SELECTION.items():
    print(f"{category[0]:10} | {category[1]:12} | {count}")

print("\nTotal selected:", len(selected))
print("Saved:", OUTPUT)
