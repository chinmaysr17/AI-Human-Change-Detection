"""
CNN Model for Human-Induced Change Detection
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_change_detection_model(input_shape=(256, 256, 6)):
    """
    Build a basic CNN for satellite image change detection.

    Input:
        Two RGB images combined together:
        3 channels (Image A) + 3 channels (Image B) = 6 channels

    Output:
        Binary change map
    """

    inputs = layers.Input(shape=input_shape)

    # Feature extraction
    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    # Restore spatial resolution
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.UpSampling2D((2, 2))(x)

    # Binary change map
    outputs = layers.Conv2D(
        1,
        (1, 1),
        padding="same",
        activation="sigmoid"
    )(x)

    model = models.Model(inputs, outputs)

    return model