import tensorflow as tf
from tensorflow.keras import layers, models


def convolution_block(inputs, filters, dropout_rate=0.0):

    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal"
    )(inputs)

    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = layers.Activation("relu")(x)

    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)

    return x


def build_mining_unet_v2(input_shape=(256, 256, 4)):

    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = convolution_block(inputs, 16)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = convolution_block(p1, 32, 0.05)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = convolution_block(p2, 64, 0.10)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = convolution_block(p3, 128, 0.20)

    # Decoder
    u3 = layers.Conv2DTranspose(
        64, 2, strides=2, padding="same"
    )(c4)

    u3 = layers.Concatenate()([u3, c3])
    c5 = convolution_block(u3, 64, 0.10)

    u2 = layers.Conv2DTranspose(
        32, 2, strides=2, padding="same"
    )(c5)

    u2 = layers.Concatenate()([u2, c2])
    c6 = convolution_block(u2, 32, 0.05)

    u1 = layers.Conv2DTranspose(
        16, 2, strides=2, padding="same"
    )(c6)

    u1 = layers.Concatenate()([u1, c1])
    c7 = convolution_block(u1, 16)

    outputs = layers.Conv2D(
        1, 1, activation="sigmoid", padding="same"
    )(c7)

    return models.Model(
        inputs,
        outputs,
        name="Mining_UNet_V2"
    )


if __name__ == "__main__":

    model = build_mining_unet_v2()

    model.summary()

    sample = tf.zeros((1, 256, 256, 4))
    prediction = model(sample, training=False)

    print("\nV2 FORWARD PASS")
    print("Input:", sample.shape)
    print("Output:", prediction.shape)
    print("Parameters:", model.count_params())
