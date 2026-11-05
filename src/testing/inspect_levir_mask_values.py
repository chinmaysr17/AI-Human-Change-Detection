
import os
import numpy as np
from PIL import Image

LABEL_DIR = "dataset/LEVIR-CD/label"
EXPECTED_VALUES = {0, 255}

exception_count = 0

print("=" * 60)
print("LEVIR-CD EXCEPTIONAL MASK PIXEL INSPECTION")
print("=" * 60)

for filename in sorted(os.listdir(LABEL_DIR)):
    if not filename.lower().endswith(".png"):
        continue

    path = os.path.join(LABEL_DIR, filename)

    with Image.open(path) as image:
        mask = np.array(image)

    values, counts = np.unique(mask, return_counts=True)

    unusual = [
        (int(value), int(count))
        for value, count in zip(values, counts)
        if value not in EXPECTED_VALUES
    ]

    if unusual:
        exception_count += sum(count for _, count in unusual)

        print(f"\nFile: {filename}")

        for value, count in unusual:
            coordinates = np.argwhere(mask == value)

            print(f"Value {value}: {count} pixels")
            print("Coordinates (row, column):")
            print(coordinates[:10].tolist())

print("\n" + "=" * 60)
print("Total exceptional pixels:", exception_count)
print("=" * 60)
