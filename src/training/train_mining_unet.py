from pathlib import Path
import tensorflow as tf

from src.training.mining_data_loader import load_mining_split
from src.model.mining_unet import build_mining_unet


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "src" / "training" / "models"

BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 0.0001
SEED = 42

tf.keras.utils.set_random_seed(SEED)


def dice_loss(y_true, y_pred):
    smooth = 1.0

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    dice = (2.0 * intersection + smooth) / (
        denominator + smooth
    )

    return 1.0 - dice


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(
        y_true, y_pred
    )

    return tf.reduce_mean(bce) + dice_loss(
        y_true, y_pred
    )


def mean_iou(y_true, y_pred):
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(
        y_true + y_pred - y_true * y_pred
    )

    return (intersection + 1e-7) / (union + 1e-7)


def precision_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)

    return tp / (tp + fp + 1e-7)


def recall_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))

    return tp / (tp + fn + 1e-7)


def augment_image(image, mask):
    """
    Apply identical spatial transformations
    to the satellite image and its segmentation mask.
    """

    # Horizontal flip
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

    # Vertical flip
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

    # Random rotation by 0, 90, 180 or 270 degrees
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
            buffer_size=len(images),
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

    print("=" * 60)
    print("MINING U-NET TRAINING")
    print("=" * 60)

    print("\nTraining device: CPU")
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", EPOCHS)

    # Load training and validation data.
    X_train, y_train = load_mining_split("train")
    X_val, y_val = load_mining_split("validation")

    train_dataset = make_dataset(
        X_train, y_train, training=True
    )

    validation_dataset = make_dataset(
        X_val, y_val, training=False
    )

    # Build separate mining U-Net.
    print("\nBuilding mining U-Net...")

    model = build_mining_unet()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=combined_loss,
        metrics=[
            mean_iou,
            precision_metric,
            recall_metric
        ]
    )

    model.summary()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_model_path = MODEL_DIR / "mining_unet_best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
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
            str(MODEL_DIR / "mining_training_history.csv")
        )
    ]

    print("\nStarting training...")
    print("=" * 60)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model with best restored weights.
    final_model_path = MODEL_DIR / "mining_unet_final.keras"
    model.save(final_model_path)

    print("\n" + "=" * 60)
    print("MINING U-NET TRAINING COMPLETED")
    print("=" * 60)

    print("Best checkpoint:", best_model_path)
    print("Final model:", final_model_path)
    print("Training history:", MODEL_DIR / "mining_training_history.csv")

    print("\nCompleted epochs:", len(history.history["loss"]))

    print(
        "Final training loss:",
        history.history["loss"][-1]
    )

    print(
        "Final validation loss:",
        history.history["val_loss"][-1]
    )


if __name__ == "__main__":
    main()
