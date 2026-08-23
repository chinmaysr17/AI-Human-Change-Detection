"""
Training and Validation Dataset Split
"""

import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(images, labels, test_size=0.2, random_state=42):
    """
    Split images and labels into training and validation sets.

    80% of the data is used for training.
    20% of the data is used for validation.
    """

    X_train, X_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    print("\n========================================")
    print("Dataset Split")
    print("========================================")

    print("Training images :", X_train.shape)
    print("Training labels :", y_train.shape)

    print("Validation images:", X_val.shape)
    print("Validation labels:", y_val.shape)

    return X_train, X_val, y_train, y_val