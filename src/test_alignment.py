
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

# src/test_alignment.py -> project root
BASE_DIR = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    BASE_DIR
    / "data"
    / "sentinel2"
    / "analyses"
    / "analysis_acquisition_test_01"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "change_detection_cnn_final.keras"
)

IMAGE_SIZE = (256, 256)


# ==================================================
# IMAGE LOADING
# ==================================================

def load_image(image_path):

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    if not np.isfinite(image).all():
        raise ValueError(
            f"Invalid image values: {image_path}"
        )

    return image


# ==================================================
# BOUNDARY MASK
# ==================================================

def create_boundary_mask():

    border = np.zeros(
        IMAGE_SIZE,
        dtype=bool
    )

    border[:5, :] = True
    border[-5:, :] = True
    border[:, :5] = True
    border[:, -5:] = True

    return border


# ==================================================
# CNN PREDICTION COMPARISON
# ==================================================

def test_prediction(
    model,
    before,
    after,
    name,
    border
):

    combined = np.concatenate(
        [before, after],
        axis=-1
    )

    model_input = np.expand_dims(
        combined,
        axis=0
    )

    prediction = model.predict(
        model_input,
        verbose=0
    )[0, :, :, 0]

    if not np.isfinite(prediction).all():
        raise ValueError(
            f"Invalid prediction for {name}"
        )

    interior = ~border

    print(f"\n{name}")
    print("=" * 45)

    print(
        "Maximum:",
        float(prediction.max())
    )

    print(
        "Minimum:",
        float(prediction.min())
    )

    print(
        "Mean:",
        float(prediction.mean())
    )

    print(
        "Boundary mean:",
        float(prediction[border].mean())
    )

    print(
        "Interior mean:",
        float(prediction[interior].mean())
    )

    print(
        "Boundary pixels >= 0.05:",
        int(np.sum(prediction[border] >= 0.05))
    )

    print(
        "Interior pixels >= 0.05:",
        int(np.sum(prediction[interior] >= 0.05))
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n===================================")
    print(" SENTINEL-2 ALIGNMENT COMPARISON")
    print("===================================")

    print("\nProject directory:", BASE_DIR)
    print("Image directory:", IMAGE_DIR)
    print("Model path:", MODEL_PATH)

    # Validate paths.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    print("\nLoading satellite images...")

    before = load_image(
        IMAGE_DIR / "before.png"
    )

    after_original = load_image(
        IMAGE_DIR / "after.png"
    )

    after_aligned = load_image(
        IMAGE_DIR / "after_aligned.png"
    )

    print("All images loaded successfully.")

    print("Before shape:", before.shape)
    print("Original after shape:", after_original.shape)
    print("Aligned after shape:", after_aligned.shape)

    # Validate model input.
    print("\nLoading CNN model...")

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    print("CNN model loaded successfully.")
    print("Model input:", model.input_shape)
    print("Model output:", model.output_shape)

    if model.input_shape[-1] != 6:
        raise ValueError(
            "Expected CNN input with 6 channels."
        )

    border = create_boundary_mask()

    # Test A: Original after image.
    test_prediction(
        model=model,
        before=before,
        after=after_original,
        name="ORIGINAL AFTER IMAGE",
        border=border
    )

    # Test B: Geometrically aligned after image.
    test_prediction(
        model=model,
        before=before,
        after=after_aligned,
        name="ALIGNED AFTER IMAGE",
        border=border
    )

    print("\n===================================")
    print(" ALIGNMENT COMPARISON COMPLETED")
    print("===================================")

    print(
        "\nNOTE: This comparison investigates "
        "prediction differences only. It does not "
        "establish Sentinel-2 detection accuracy."
    )


# ==================================================
# EXECUTION
# ==================================================

if __name__ == "__main__":
    main()
