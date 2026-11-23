import tensorflow as tf
from tensorflow.keras import layers, models


def convolution_block(inputs, filters):
    """Two convolution layers with batch normalization."""

    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def build_mining_unet(input_shape=(256, 256, 4)):
    """
    Build a lightweight U-Net for mining-area segmentation.

    Input:
        Four Sentinel-2 bands: B02, B03, B04, B08.

    Output:
        Binary mining segmentation mask.
    """

    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = convolution_block(inputs, 16)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = convolution_block(p1, 32)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = convolution_block(p2, 64)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = convolution_block(p3, 128)
    c4 = layers.Dropout(0.3)(c4)

    # Decoder
    u3 = layers.Conv2DTranspose(
        64, 2, strides=2, padding="same"
    )(c4)

    u3 = layers.Concatenate()([u3, c3])
    c5 = convolution_block(u3, 64)

    u2 = layers.Conv2DTranspose(
        32, 2, strides=2, padding="same"
    )(c5)

    u2 = layers.Concatenate()([u2, c2])
    c6 = convolution_block(u2, 32)

    u1 = layers.Conv2DTranspose(
        16, 2, strides=2, padding="same"
    )(c6)

    u1 = layers.Concatenate()([u1, c1])
    c7 = convolution_block(u1, 16)

    # Binary segmentation output
    outputs = layers.Conv2D(
        1, 1, activation="sigmoid", padding="same"
    )(c7)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Mining_UNet"
    )

    return model


if __name__ == "__main__":

    model = build_mining_unet()

    print("\nMINING U-NET MODEL")
    print("=" * 55)

    model.summary()

    # Test a dummy four-band input
    sample = tf.zeros((1, 256, 256, 4))
    prediction = model(sample, training=False)

    print("\nForward Pass Test")
    print("Input shape:", sample.shape)
    print("Output shape:", prediction.shape)
    print("Output minimum:", float(tf.reduce_min(prediction)))
    print("Output maximum:", float(tf.reduce_max(prediction)))
