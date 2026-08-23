"""
Test Train / Validation / Test Split
"""

import os
import sys

# Add src directory to Python path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from training.data_loader import load_dataset
from training.split_train_val_test import split_train_val_test


def main():

    print("=" * 60)
    print("TESTING THREE-WAY DATASET SPLIT")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    images, labels = load_dataset()

    print("✅ Dataset loaded.")

    print(
        "Images shape:",
        images.shape
    )

    print(
        "Labels shape:",
        labels.shape
    )

    # --------------------------------------------------
    # 2. Split dataset
    # --------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_train_val_test(
        images,
        labels,
        test_size=0.10,
        validation_size=0.10,
        random_state=42
    )

    # --------------------------------------------------
    # 3. Final verification
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST SPLIT VERIFICATION")
    print("=" * 60)

    print(
        "\nTraining samples   :",
        len(X_train)
    )

    print(
        "Validation samples :",
        len(X_val)
    )

    print(
        "Test samples       :",
        len(X_test)
    )

    total = (
        len(X_train)
        + len(X_val)
        + len(X_test)
    )

    print(
        "Total samples      :",
        total
    )

    print("\nShapes:")

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_val  :",
        X_val.shape
    )

    print(
        "X_test :",
        X_test.shape
    )

    print(
        "\ny_train:",
        y_train.shape
    )

    print(
        "y_val  :",
        y_val.shape
    )

    print(
        "y_test :",
        y_test.shape
    )

    # --------------------------------------------------
    # 4. Check result
    # --------------------------------------------------

    if (
        len(X_train) == 102
        and len(X_val) == 13
        and len(X_test) == 13
        and total == len(images)
    ):

        print(
            "\n✅ Three-way dataset split verified successfully."
        )

    else:

        print(
            "\n❌ Three-way dataset split verification failed."
        )


if __name__ == "__main__":
    main()