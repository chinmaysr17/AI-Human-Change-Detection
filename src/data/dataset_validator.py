"""
Dataset Validator
"""

import os
import config


def validate_dataset():

    print("\nChecking LEVIR-CD Dataset...\n")

    print("DATASET_DIR :", config.DATASET_DIR)
    print("LEVIR_DATASET :", config.LEVIR_DATASET)
    print("IMAGE_A_DIR :", config.IMAGE_A_DIR)
    print("IMAGE_B_DIR :", config.IMAGE_B_DIR)
    print("LABEL_DIR :", config.LABEL_DIR)
    print("-" * 60)

    folders = {
        "Image A": config.IMAGE_A_DIR,
        "Image B": config.IMAGE_B_DIR,
        "Label": config.LABEL_DIR
    }

    for name, path in folders.items():

        if os.path.exists(path):
            print(f"✅ {name} folder found.")
        else:
            print(f"❌ {name} folder NOT found.")
            return False

    a_count = len(os.listdir(config.IMAGE_A_DIR))
    b_count = len(os.listdir(config.IMAGE_B_DIR))
    label_count = len(os.listdir(config.LABEL_DIR))

    print("\nImage Count")
    print("---------------------------")
    print(f"Image A : {a_count}")
    print(f"Image B : {b_count}")
    print(f"Label   : {label_count}")

    if a_count == b_count == label_count:
        print("\n✅ Dataset validation successful.")
        return True

    print("\n❌ Dataset is inconsistent.")
    return False