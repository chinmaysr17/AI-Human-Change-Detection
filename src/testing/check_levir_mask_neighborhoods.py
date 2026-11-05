
import os
import numpy as np
from PIL import Image

LABEL_DIR = "dataset/LEVIR-CD/label"

TARGETS = {
    "test_18.png": 254,
    "test_68.png": 156,
    "test_70.png": 156,
    "test_73.png": 156,
    "test_74.png": 156,
}

for filename, target_value in TARGETS.items():
    path = os.path.join(LABEL_DIR, filename)

    with Image.open(path) as image:
        mask = np.array(image)

    coordinates = np.argwhere(mask == target_value)

    print("\n" + "=" * 55)
    print(f"{filename} — value {target_value}")
    print("=" * 55)

    for row, col in coordinates[:5]:
        r1 = max(0, row - 2)
        r2 = min(mask.shape[0], row + 3)
        c1 = max(0, col - 2)
        c2 = min(mask.shape[1], col + 3)

        neighborhood = mask[r1:r2, c1:c2]

        print(f"\nPixel location: ({row}, {col})")
        print(neighborhood.tolist())
