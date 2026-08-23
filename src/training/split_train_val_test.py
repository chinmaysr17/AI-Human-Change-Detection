"""
Train / Validation / Test Dataset Split
"""

import os
import sys

# Add src directory to Python path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from sklearn.model_selection import train_test_split


def split_train_val_test(
    images,
    labels,
    test_size=0.10,
    validation_size=0.10,
    random_state=42
):
    """
    Split dataset into training, validation and test sets.

    Default:
        80% Training
        10% Validation
        10% Test
    """

    # --------------------------------------------------
    # 1. First split
    # --------------------------------------------------
    # Separate final test set first.
    # This test set will remain completely independent.

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # --------------------------------------------------
    # 2. Calculate validation proportion
    # --------------------------------------------------

    validation_ratio = (
        validation_size /
        (1 - test_size)
    )

    # --------------------------------------------------
    # 3. Split remaining data into
    #    training and validation
    # --------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_ratio,
        random_state=random_state,
        shuffle=True
    )

    # --------------------------------------------------
    # 4. Display split information
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 60)

    print(
        "\nTotal images      :",
        len(images)
    )

    print(
        "Training images   :",
        len(X_train)
    )

    print(
        "Validation images :",
        len(X_val)
    )

    print(
        "Test images       :",
        len(X_test)
    )

    print(
        "\nTraining shape:",
        X_train.shape
    )

    print(
        "Training labels:",
        y_train.shape
    )

    print(
        "Validation shape:",
        X_val.shape
    )

    print(
        "Validation labels:",
        y_val.shape
    )

    print(
        "Test shape:",
        X_test.shape
    )

    print(
        "Test labels:",
        y_test.shape
    )

    # --------------------------------------------------
    # 5. Verify total samples
    # --------------------------------------------------

    total_samples = (
        len(X_train)
        + len(X_val)
        + len(X_test)
    )

    print("\n" + "-" * 60)
    print("FINAL SPLIT VERIFICATION")
    print("-" * 60)

    print(
        "Training samples   :",
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

    print(
        "Total samples      :",
        total_samples
    )

    if total_samples == len(images):

        print(
            "\n✅ Three-way dataset split successful."
        )

    else:

        print(
            "\n❌ Dataset split verification failed."
        )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


if __name__ == "__main__":

    # This section is only a basic module check.
    print(
        "✅ split_train_val_test.py loaded successfully."
    )