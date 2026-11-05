
import os
from collections import Counter

import numpy as np
from PIL import Image

ROOT = "dataset/LEVIR-CD"

FOLDERS = {
    "before": "A",
    "after": "B",
    "label": "label"
}


def get_png_files(folder):
    path = os.path.join(ROOT, folder)
    return {
        os.path.splitext(name)[0]: os.path.join(path, name)
        for name in os.listdir(path)
        if name.lower().endswith(".png")
    }


def main():
    print("=" * 60)
    print("LEVIR-CD DATASET AUDIT")
    print("=" * 60)

    files = {
        key: get_png_files(folder)
        for key, folder in FOLDERS.items()
    }

    for key, items in files.items():
        print(f"{key}: {len(items)} images")

    common_names = (
        set(files["before"])
        & set(files["after"])
        & set(files["label"])
    )

    all_names = (
        set(files["before"])
        | set(files["after"])
        | set(files["label"])
    )

    missing = sorted(all_names - common_names)

    print("\nMatching image-mask triplets:", len(common_names))
    print("Incomplete triplets:", len(missing))

    dimension_errors = []
    mode_summary = Counter()
    mask_values = Counter()
    sample_mask_values = {}

    for name in sorted(common_names):
        paths = [
            files["before"][name],
            files["after"][name],
            files["label"][name]
        ]

        images = [Image.open(path) for path in paths]

        sizes = [image.size for image in images]
        modes = [image.mode for image in images]

        if len(set(sizes)) != 1:
            dimension_errors.append(name)

        for mode in modes:
            mode_summary[mode] += 1

        mask = np.array(images[2])
        unique, counts = np.unique(mask, return_counts=True)

        for value, count in zip(unique, counts):
            mask_values[int(value)] += int(count)

        if len(sample_mask_values) < 5:
            sample_mask_values[name] = unique.tolist()

        for image in images:
            image.close()

    print("\nDimension mismatches:", len(dimension_errors))
    print("Image modes:", dict(mode_summary))
    print("Mask pixel values and counts:", dict(mask_values))
    print("Sample mask encodings:", sample_mask_values)

    print("\n" + "=" * 60)

    if not missing and not dimension_errors:
        print("STRUCTURAL VALIDATION: PASS")
    else:
        print("STRUCTURAL VALIDATION: NEEDS ATTENTION")

    print("=" * 60)


if __name__ == "__main__":
    main()
