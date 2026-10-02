import os
import sys

import numpy as np
from tensorflow.keras.models import load_model


# ============================================================
# PROJECT PATH SETUP
# ============================================================

SRC_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# EXISTING PROJECT MODULES
# ============================================================

import config

from data.image_loader import load_image_pair
from preprocessing.image_loader import preprocess_images


# ============================================================
# CHANGE DETECTION FUNCTION
# ============================================================

def run_change_detection(filename):
    """
    Run the trained CNN model on a satellite image pair.

    Parameters:
        filename (str):
            Satellite image filename, for example:
            test_1.png

    Returns:
        Dictionary containing prediction results.
    """

    filename = filename.strip()

    # --------------------------------------------------------
    # 1. Load image pair
    # --------------------------------------------------------

    image_a, image_b, label = load_image_pair(filename)

    if (
        image_a is None
        or image_b is None
        or label is None
    ):
        raise ValueError(
            "Image pair could not be loaded."
        )

    # --------------------------------------------------------
    # 2. Validate image dimensions
    # --------------------------------------------------------

    if image_a.shape[:2] != image_b.shape[:2]:
        raise ValueError(
            "Image A and Image B dimensions do not match."
        )

    if image_a.shape[:2] != label.shape[:2]:
        raise ValueError(
            "Image and label dimensions do not match."
        )

    # --------------------------------------------------------
    # 3. Preprocess images
    # --------------------------------------------------------

    processed_a, processed_b, processed_label = (
        preprocess_images(
            image_a,
            image_b,
            label
        )
    )

    # --------------------------------------------------------
    # 4. Create 6-channel input
    # --------------------------------------------------------

    combined_image = np.concatenate(
        [
            processed_a,
            processed_b
        ],
        axis=-1
    )

    if combined_image.shape[-1] != 6:
        raise ValueError(
            "Expected 6-channel input."
        )

    # Add batch dimension
    model_input = np.expand_dims(
        combined_image,
        axis=0
    )

    # --------------------------------------------------------
    # 5. Load trained CNN model
    # --------------------------------------------------------

    model_path = os.path.join(
        config.BASE_DIR,
        "models",
        "change_detection_cnn_final.keras"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = load_model(
        model_path,
        compile=False
    )

    # --------------------------------------------------------
    # 6. Generate prediction
    # --------------------------------------------------------

    prediction = model.predict(
        model_input,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # 7. Convert prediction into binary change map
    # --------------------------------------------------------

    threshold = 0.50

    predicted_change = (
        prediction >= threshold
    ).astype(np.uint8)

    predicted_change = (
        predicted_change.squeeze()
    )

    # --------------------------------------------------------
    # 8. Calculate change statistics
    # --------------------------------------------------------

    predicted_pixels = int(
        np.sum(
            predicted_change == 1
        )
    )

    total_pixels = int(
        predicted_change.size
    )

    change_percentage = (
        predicted_pixels
        / total_pixels
    ) * 100

    # --------------------------------------------------------
    # 9. Return results
    # --------------------------------------------------------

    return {
        "filename": filename,
        "threshold": threshold,
        "prediction_shape": prediction.shape,
        "predicted_change_pixels": predicted_pixels,
        "total_pixels": total_pixels,
        "change_percentage": change_percentage,
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "prediction_mean": float(prediction.mean()),
        "change_map": predicted_change
    }