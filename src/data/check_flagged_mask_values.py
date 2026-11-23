
from pathlib import Path
import numpy as np

DATA_DIR = Path("dataset/Mining/training_chips_30")

TILE_IDS = [2120, 2987, 47, 783]

for tile_id in TILE_IDS:
    mask_path = DATA_DIR / f"mask_{tile_id}.npy"
    mask = np.load(mask_path)

    values, counts = np.unique(mask, return_counts=True)

    print(f"\nTILE {tile_id}")
    print("-" * 35)
    print("Shape:", mask.shape)
    print("Data type:", mask.dtype)
    print("Minimum:", mask.min())
    print("Maximum:", mask.max())
    print("Unique values:", values)
    print("Value counts:", dict(zip(values.tolist(), counts.tolist())))
    print("Positive pixels:", np.count_nonzero(mask > 0))
    print("Coverage:", round(np.mean(mask > 0) * 100, 2), "%")
