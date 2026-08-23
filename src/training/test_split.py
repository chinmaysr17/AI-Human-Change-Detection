"""
Test Training and Validation Split
"""

import os
import sys

# Add src directory to Python path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from training.data_loader import load_dataset
from training.split_dataset import split_dataset


def main():

    print("=" * 60)
    print("TRAINING / VALIDATION SPLIT TEST")
    print("=" * 60)

    # Load complete dataset
    images, labels = load_dataset()

    # Split dataset
    X_train, X_val, y_train, y_val = split_dataset(
        images,
        labels,
        test_size=0.2,
        random_state=42
    )

    print("\nFinal Split Verification")
    print("----------------------------")

    print("Training samples  :", len(X_train))
    print("Validation samples:", len(X_val))

    total_samples = len(X_train) + len(X_val)

    print("Total samples     :", total_samples)

    if total_samples == len(images):
        print("\n✅ Dataset split successful.")
    else:
        print("\n❌ Dataset split error.")


if __name__ == "__main__":
    main()