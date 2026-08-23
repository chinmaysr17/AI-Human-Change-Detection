"""
Improved CNN Training Script
"""

import os
import sys
import tensorflow as tf

# Add src directory to Python path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import config

from training.data_loader import load_dataset
from training.split_dataset import split_dataset
from model.change_detection_cnn import build_change_detection_model


def weighted_binary_crossentropy(pos_weight=3.0):
    """
    Weighted Binary Cross-Entropy Loss.

    Gives more importance to changed pixels because the
    LEVIR-CD dataset contains many more unchanged pixels.
    """

    def loss(y_true, y_pred):

        # Avoid log(0)
        epsilon = tf.keras.backend.epsilon()

        y_pred = tf.clip_by_value(
            y_pred,
            epsilon,
            1.0 - epsilon
        )

        loss_value = -(
            pos_weight * y_true * tf.math.log(y_pred)
            +
            (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

        return tf.reduce_mean(loss_value)

    return loss


def main():

    print("=" * 60)
    print("IMPROVED CNN TRAINING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    images, labels = load_dataset()

    print("✅ Dataset loaded.")

    # --------------------------------------------------
    # 2. Split dataset
    # --------------------------------------------------

    print("\nSplitting dataset...")

    X_train, X_val, y_train, y_val = split_dataset(
        images,
        labels,
        test_size=0.2,
        random_state=42
    )

    print("✅ Dataset split completed.")

    # --------------------------------------------------
    # 3. Build CNN model
    # --------------------------------------------------

    print("\nBuilding CNN model...")

    model = build_change_detection_model()

    print("✅ CNN model created.")

    # --------------------------------------------------
    # 4. Configure optimizer
    # --------------------------------------------------

    print("\nConfiguring optimizer...")

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE
    )

    print(
        "Learning rate:",
        config.LEARNING_RATE
    )

    # --------------------------------------------------
    # 5. Compile model
    # --------------------------------------------------

    print("\nCompiling improved model...")

    model.compile(
        optimizer=optimizer,
        loss=weighted_binary_crossentropy(
            pos_weight=3.0
        ),
        metrics=["accuracy"]
    )

    print("✅ Improved model compiled.")

    # --------------------------------------------------
    # 6. Display model information
    # --------------------------------------------------

    print("\nModel Summary")
    print("=" * 60)

    model.summary()

    # --------------------------------------------------
    # 7. Train improved model
    # --------------------------------------------------

    print("\nStarting improved training...")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )

    # --------------------------------------------------
    # 8. Save improved model
    # --------------------------------------------------

    models_dir = os.path.join(
        config.BASE_DIR,
        "models"
    )

    os.makedirs(
        models_dir,
        exist_ok=True
    )

    model_path = os.path.join(
        models_dir,
        "change_detection_cnn_improved.keras"
    )

    model.save(model_path)

    print("\n✅ Improved model saved successfully.")
    print("Model path:", model_path)

    # --------------------------------------------------
    # 9. Display final results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("IMPROVED TRAINING COMPLETED")
    print("=" * 60)

    print(
        "\nFinal Training Accuracy:",
        history.history["accuracy"][-1]
    )

    print(
        "Final Validation Accuracy:",
        history.history["val_accuracy"][-1]
    )

    print(
        "Final Training Loss:",
        history.history["loss"][-1]
    )

    print(
        "Final Validation Loss:",
        history.history["val_loss"][-1]
    )

    print(
        "\n✅ Improved CNN training completed successfully."
    )


if __name__ == "__main__":
    main()