import base64
import io

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from src.inference.change_detector import run_change_detection
from src.data.image_loader import load_image_pair


# ============================================================
# API RESPONSE MODEL
# ============================================================

class AnalysisResult(BaseModel):
    filename: str
    threshold: float
    predicted_change_pixels: int
    total_pixels: int
    change_percentage: float
    prediction_min: float
    prediction_max: float
    prediction_mean: float
    before_image: str
    after_image: str
    change_map: str
    change_overlay: str


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Human-Induced Change Detection API",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AI Human-Induced Change Detection API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# ============================================================
# CONVERT IMAGE TO BASE64
# ============================================================

def create_image_data_url(image_array):
    """
    Convert a NumPy image array into a PNG Base64 data URL.
    """

    image_array = np.asarray(image_array)

    if np.issubdtype(image_array.dtype, np.floating):

        if image_array.max() <= 1.0:
            image_array = image_array * 255.0

        image_array = np.clip(
            image_array,
            0,
            255
        ).astype(np.uint8)

    elif image_array.dtype != np.uint8:

        image_array = np.clip(
            image_array,
            0,
            255
        ).astype(np.uint8)

    if (
        image_array.ndim == 3
        and image_array.shape[-1] == 1
    ):
        image_array = image_array.squeeze(-1)

    image = Image.fromarray(
        image_array
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded_image
    )


# ============================================================
# CREATE CHANGE MAP
# ============================================================

def create_change_map(change_mask):
    """
    Convert binary change mask into a PNG Base64 data URL.
    """

    change_mask = np.asarray(
        change_mask,
        dtype=np.uint8
    )

    image_array = change_mask * 255

    image = Image.fromarray(
        image_array,
        mode="L"
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded_image
    )


# ============================================================
# CREATE SATELLITE CHANGE OVERLAY
# ============================================================

def create_change_overlay(
    after_image,
    change_mask
):
    """
    Highlight detected change regions on the
    original After satellite image.

    Detected regions are shown with a red overlay.
    """

    after_array = np.asarray(
        after_image
    )

    # --------------------------------------------------------
    # Convert image to uint8
    # --------------------------------------------------------

    if np.issubdtype(
        after_array.dtype,
        np.floating
    ):

        if after_array.max() <= 1.0:
            after_array = after_array * 255.0

        after_array = np.clip(
            after_array,
            0,
            255
        ).astype(np.uint8)

    elif after_array.dtype != np.uint8:

        after_array = np.clip(
            after_array,
            0,
            255
        ).astype(np.uint8)


    # --------------------------------------------------------
    # Ensure RGB image
    # --------------------------------------------------------

    if after_array.ndim == 2:

        after_image_pil = Image.fromarray(
            after_array,
            mode="L"
        ).convert("RGB")

    else:

        after_image_pil = Image.fromarray(
            after_array
        ).convert("RGB")


    # --------------------------------------------------------
    # Resize change mask to original image size
    # --------------------------------------------------------

    mask_image = Image.fromarray(
        (
            np.asarray(
                change_mask,
                dtype=np.uint8
            ) * 255
        ),
        mode="L"
    )

    mask_image = mask_image.resize(
        after_image_pil.size,
        Image.Resampling.NEAREST
    )


    # --------------------------------------------------------
    # Create red highlight layer
    # --------------------------------------------------------

    red_layer = Image.new(
        "RGB",
        after_image_pil.size,
        (255, 0, 0)
    )


    # --------------------------------------------------------
    # Blend red with satellite image
    # --------------------------------------------------------

    highlighted_image = Image.blend(
        after_image_pil,
        red_layer,
        0.45
    )


    # --------------------------------------------------------
    # Apply highlight ONLY to changed regions
    # --------------------------------------------------------

    overlay = Image.composite(
        highlighted_image,
        after_image_pil,
        mask_image
    )


    # --------------------------------------------------------
    # Convert overlay to Base64
    # --------------------------------------------------------

    buffer = io.BytesIO()

    overlay.save(
        buffer,
        format="PNG"
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded_image
    )


# ============================================================
# AI ANALYSIS ENDPOINT
# ============================================================

@app.get(
    "/api/analyze/{filename}",
    response_model=AnalysisResult
)
def analyze_image(filename: str):

    try:

        # ----------------------------------------------------
        # Load original satellite images
        # ----------------------------------------------------

        image_a, image_b, label = load_image_pair(
            filename
        )

        if (
            image_a is None
            or image_b is None
        ):
            raise ValueError(
                "Satellite image pair could not be loaded."
            )


        # ----------------------------------------------------
        # Run CNN change detection
        # ----------------------------------------------------

        result = run_change_detection(
            filename
        )


        # ----------------------------------------------------
        # Create visual outputs
        # ----------------------------------------------------

        before_image = create_image_data_url(
            image_a
        )

        after_image = create_image_data_url(
            image_b
        )

        change_map = create_change_map(
            result["change_map"]
        )

        change_overlay = create_change_overlay(
            image_b,
            result["change_map"]
        )


        # ----------------------------------------------------
        # Return complete result
        # ----------------------------------------------------

        return {
            "filename": result["filename"],
            "threshold": result["threshold"],
            "predicted_change_pixels": result[
                "predicted_change_pixels"
            ],
            "total_pixels": result[
                "total_pixels"
            ],
            "change_percentage": result[
                "change_percentage"
            ],
            "prediction_min": result[
                "prediction_min"
            ],
            "prediction_max": result[
                "prediction_max"
            ],
            "prediction_mean": result[
                "prediction_mean"
            ],
            "before_image": before_image,
            "after_image": after_image,
            "change_map": change_map,
            "change_overlay": change_overlay
        }


    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )


    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"AI processing failed: {str(error)}"
        )