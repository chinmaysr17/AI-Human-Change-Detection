"""
Final CNN Training Script
Uses:
102 Training Samples
13 Validation Samples
13 Final Test Samples
"""

import os
import sys
import tensorflow as tf

# Add src directory to Python path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import config

from training.data_loader import load_dataset
from training.split_train_val_test import split_train_val_test
from model.change_detection_cnn import build_change_detection_model


def weighted_binary_crossentropy(pos_weight=3.0):
    """
    Weighted Binary Cross-Entropy Loss.

    Gives more importance to changed pixels because
    unchanged pixels are much more common in the dataset.
    """

    def weighted_loss(y_true, y_pred):

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

    return weighted_loss


def main():

    print("=" * 60)
    print("FINAL CNN TRAINING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load complete dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    images, labels = load_dataset()

    print("✅ Dataset loaded.")

    print(
        "Images:",
        images.shape
    )

    print(
        "Labels:",
        labels.shape
    )

    # --------------------------------------------------
    # 2. Three-way dataset split
    # --------------------------------------------------

    print("\nSplitting dataset...")

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

    print("\n✅ Three-way split completed.")

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_val)
    )

    print(
        "Final test samples:",
        len(X_test)
    )

    # --------------------------------------------------
    # 3. Build CNN
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

    print("\nCompiling final model...")

    model.compile(
        optimizer=optimizer,
        loss=weighted_binary_crossentropy(
            pos_weight=3.0
        ),
        metrics=["accuracy"]
    )

    print("✅ Final model compiled.")

    # --------------------------------------------------
    # 6. Display model
    # --------------------------------------------------

    print("\nModel Summary")
    print("=" * 60)

    model.summary()

    # --------------------------------------------------
    # 7. Train model
    # --------------------------------------------------

    print("\nStarting final training...")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )

    # --------------------------------------------------
    # 8. Save final model
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
        "change_detection_cnn_final.keras"
    )

    model.save(model_path)

    print(
        "\n✅ Final CNN model saved successfully."
    )

    print(
        "Model path:",
        model_path
    )

    # --------------------------------------------------
    # 9. Final training results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TRAINING COMPLETED")
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

    # --------------------------------------------------
    # 10. Verify test data was NOT used for training
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST DATA PROTECTION")
    print("=" * 60)

    print(
        "\nFinal test samples reserved:",
        len(X_test)
    )

    print(
        "These samples were NOT used during training."
    )

    print(
        "\n✅ Final test set successfully kept separate."
    )

    print(
        "\n✅ Final CNN training completed successfully."
    )


if __name__ == "__main__":
    main()