from pathlib import Path
import numpy as np
import tensorflow as tf

from src.training.mining_data_loader_v2 import (
    load_mining_split,
    calculate_band_statistics,
    standardize_images
)

from src.model.mining_unet_v2 import build_mining_unet_v2


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "src" / "training" / "models"

BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 0.0001
SEED = 42

# Focal Tversky parameters
ALPHA = 0.3
BETA = 0.7
GAMMA = 0.75
SMOOTH = 1.0

tf.keras.utils.set_random_seed(SEED)


def combined_focal_tversky_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Binary Cross-Entropy
    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )

    # Tversky index calculated separately for each image
    axes = [1, 2, 3]

    tp = tf.reduce_sum(y_true * y_pred, axis=axes)

    fp = tf.reduce_sum(
        (1.0 - y_true) * y_pred,
        axis=axes
    )

    fn = tf.reduce_sum(
        y_true * (1.0 - y_pred),
        axis=axes
    )

    tversky = (
        tp + SMOOTH
    ) / (
        tp + ALPHA * fp + BETA * fn + SMOOTH
    )

    focal_tversky = tf.reduce_mean(
        tf.pow(1.0 - tversky, GAMMA)
    )

    return bce + focal_tversky


def mean_iou(y_true, y_pred):

    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(
        y_true + y_pred - y_true * y_pred
    )

    return (intersection + 1e-7) / (union + 1e-7)


def precision_metric(y_true, y_pred):

    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)

    return tp / (tp + fp + 1e-7)


def recall_metric(y_true, y_pred):

    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))

    return tp / (tp + fn + 1e-7)


def augment_image(image, mask):

    flip_lr = tf.random.uniform(()) > 0.5

    image = tf.cond(
        flip_lr,
        lambda: tf.image.flip_left_right(image),
        lambda: image
    )

    mask = tf.cond(
        flip_lr,
        lambda: tf.image.flip_left_right(mask),
        lambda: mask
    )

    flip_ud = tf.random.uniform(()) > 0.5

    image = tf.cond(
        flip_ud,
        lambda: tf.image.flip_up_down(image),
        lambda: image
    )

    mask = tf.cond(
        flip_ud,
        lambda: tf.image.flip_up_down(mask),
        lambda: mask
    )

    k = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=4,
        dtype=tf.int32
    )

    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    return image, mask


def make_dataset(images, masks, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (images, masks)
    )

    if training:

        dataset = dataset.shuffle(
            len(images),
            seed=SEED,
            reshuffle_each_iteration=True
        )

        dataset = dataset.map(
            augment_image,
            num_parallel_calls=1
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1

    return dataset.with_options(options)


def main():

    print("=" * 65)
    print("MINING U-NET V3 TRAINING - FOCAL TVERSKY")
    print("=" * 65)

    X_train, y_train = load_mining_split("train")
    X_val, y_val = load_mining_split("validation")

    # Calculate normalization statistics from training data only.
    band_mean, band_std = calculate_band_statistics(X_train)

    X_train = standardize_images(
        X_train, band_mean, band_std
    )

    X_val = standardize_images(
        X_val, band_mean, band_std
    )

    print("\nStandardized training shape:", X_train.shape)
    print("Standardized validation shape:", X_val.shape)

    train_dataset = make_dataset(
        X_train, y_train, training=True
    )

    val_dataset = make_dataset(
        X_val, y_val, training=False
    )

    # Same architecture as V2.
    model = build_mining_unet_v2()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=combined_focal_tversky_loss,
        metrics=[
            mean_iou,
            precision_metric,
            recall_metric
        ]
    )

    model.summary()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_path = MODEL_DIR / "mining_unet_v3_best.keras"
    final_path = MODEL_DIR / "mining_unet_v3_final.keras"
    history_path = MODEL_DIR / "mining_training_history_v3.csv"

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            str(best_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            str(history_path)
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    model.save(final_path)

    best_epoch = int(
        np.argmin(history.history["val_loss"])
    ) + 1

    print("\n" + "=" * 65)
    print("MINING U-NET V3 TRAINING COMPLETED")
    print("=" * 65)

    print("Completed epochs:", len(history.history["loss"]))
    print("Best epoch:", best_epoch)
    print("Best validation loss:", min(history.history["val_loss"]))
    print("Final training loss:", history.history["loss"][-1])

    print("\nSaved files:")
    print(best_path)
    print(final_path)
    print(history_path)


if __name__ == "__main__":
    main()
