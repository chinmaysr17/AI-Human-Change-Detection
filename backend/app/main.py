import base64
import io
import json
import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from tensorflow.keras.models import load_model

from src.inference.change_detector import run_change_detection
from src.data.image_loader import load_image_pair
from src.data.sentinel2_service import retrieve_sentinel2_pair


# =========================================================
# API REQUEST / RESPONSE MODELS
# =========================================================

class AnalysisSelectionRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0, le=100)
    before_date: date
    after_date: date


class AnalysisSelectionResponse(BaseModel):
    status: str
    message: str
    latitude: float
    longitude: float
    radius_km: float
    before_date: date
    after_date: date
    imagery_retrieved: bool
    record_id: int
    created_at: str
    analysis_status: str


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
    change_regions: list[dict]


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="AI Human-Induced Change Detection API",
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# PROJECT PATHS AND SQLITE HISTORY
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = DATA_DIRECTORY / "analysis_history.db"

SENTINEL_MODEL_PATH = PROJECT_ROOT / "models" / "change_detection_cnn_final.keras"
SENTINEL_IMAGE_SIZE = (256, 256)
SENTINEL_THRESHOLD = 0.50
_sentinel_model = None


def get_database_connection():
    """Create a connection to the local analysis-history database."""
    connection = sqlite3.connect(str(DATABASE_PATH), timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_history_database():
    """Create the analysis history table if it does not exist."""
    with get_database_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                radius_km REAL NOT NULL,
                before_date TEXT NOT NULL,
                after_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Pending Analysis',
                imagery_retrieved INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.commit()


initialize_history_database()


# =========================================================
# BASIC ENDPOINTS
# =========================================================

@app.get("/")
def root():
    return {"message": "AI Human-Induced Change Detection API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# =========================================================
# LOCATION AND DATE SELECTION ENDPOINT
# =========================================================

@app.post(
    "/api/analysis-selection",
    response_model=AnalysisSelectionResponse,
)
def validate_analysis_selection(selection: AnalysisSelectionRequest):
    """Validate and save a location/date selection without downloading imagery."""
    if selection.after_date <= selection.before_date:
        raise HTTPException(
            status_code=422,
            detail="after_date must be later than before_date.",
        )

    if selection.before_date > date.today():
        raise HTTPException(
            status_code=422,
            detail="before_date cannot be in the future.",
        )

    if selection.after_date > date.today():
        raise HTTPException(
            status_code=422,
            detail="after_date cannot be in the future.",
        )

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with get_database_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_history (
                latitude, longitude, radius_km, before_date, after_date,
                created_at, status, imagery_retrieved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                selection.latitude,
                selection.longitude,
                selection.radius_km,
                selection.before_date.isoformat(),
                selection.after_date.isoformat(),
                created_at,
                "Pending Analysis",
                0,
            ),
        )
        record_id = cursor.lastrowid
        connection.commit()

    return {
        "status": "valid",
        "message": (
            "Location and dates validated and saved successfully. "
            "Satellite imagery has not been retrieved yet."
        ),
        "latitude": selection.latitude,
        "longitude": selection.longitude,
        "radius_km": selection.radius_km,
        "before_date": selection.before_date,
        "after_date": selection.after_date,
        "imagery_retrieved": False,
        "record_id": record_id,
        "created_at": created_at,
        "analysis_status": "Pending Analysis",
    }


# =========================================================
# ANALYSIS HISTORY ENDPOINT
# =========================================================

@app.get("/api/analysis-history")
def get_analysis_history():
    """Return saved analysis selections, newest first."""
    try:
        with get_database_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, latitude, longitude, radius_km, before_date,
                       after_date, created_at, status, imagery_retrieved
                FROM analysis_history
                ORDER BY id DESC
                """
            ).fetchall()

        return {
            "count": len(rows),
            "records": [
                {
                    "id": row["id"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "radius_km": row["radius_km"],
                    "before_date": row["before_date"],
                    "after_date": row["after_date"],
                    "created_at": row["created_at"],
                    "status": row["status"],
                    "imagery_retrieved": bool(row["imagery_retrieved"]),
                }
                for row in rows
            ],
        }
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve analysis history: {error}",
        )


# =========================================================
# MODEL METADATA
# =========================================================

MODEL_METADATA_PATH = (
    PROJECT_ROOT / "models" / "change_detection_cnn_final_metadata.json"
)


def load_model_metadata():
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(f"Model metadata not found: {MODEL_METADATA_PATH}")

    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/api/model")
def get_model_metadata():
    try:
        return load_model_metadata()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model metadata: {error}",
        )


# =========================================================
# IMAGE DISPLAY HELPERS
# =========================================================

def create_image_data_url(image_array):
    image_array = np.asarray(image_array)

    if np.issubdtype(image_array.dtype, np.floating):
        if image_array.size and image_array.max() <= 1.0:
            image_array = image_array * 255.0
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    elif image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    if image_array.ndim == 3 and image_array.shape[-1] == 1:
        image_array = image_array.squeeze(-1)

    image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded_image


def create_change_map(change_mask):
    change_mask = np.asarray(change_mask, dtype=np.uint8)
    image = Image.fromarray(change_mask * 255, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded_image


def create_change_overlay(after_image, change_mask):
    after_array = np.asarray(after_image)

    if np.issubdtype(after_array.dtype, np.floating):
        if after_array.size and after_array.max() <= 1.0:
            after_array = after_array * 255.0
        after_array = np.clip(after_array, 0, 255).astype(np.uint8)
    elif after_array.dtype != np.uint8:
        after_array = np.clip(after_array, 0, 255).astype(np.uint8)

    if after_array.ndim == 2:
        after_image_pil = Image.fromarray(after_array, mode="L").convert("RGB")
    else:
        after_image_pil = Image.fromarray(after_array).convert("RGB")

    mask_image = Image.fromarray(
        np.asarray(change_mask, dtype=np.uint8) * 255,
        mode="L",
    )
    mask_image = mask_image.resize(
        after_image_pil.size,
        Image.Resampling.NEAREST,
    )

    red_layer = Image.new("RGB", after_image_pil.size, (255, 0, 0))
    highlighted_image = Image.blend(after_image_pil, red_layer, 0.45)
    overlay = Image.composite(highlighted_image, after_image_pil, mask_image)

    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded_image


# =========================================================
# SENTINEL-2 MODEL HELPERS
# =========================================================

def get_sentinel_model():
    """Load the trained CNN once for inference; its training loss is not needed."""
    global _sentinel_model

    if _sentinel_model is None:
        if not SENTINEL_MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained CNN model not found: {SENTINEL_MODEL_PATH}")
        _sentinel_model = load_model(SENTINEL_MODEL_PATH, compile=False)

    return _sentinel_model


def load_sentinel_image(image_path):
    """Load an RGB Sentinel image and prepare a 256x256 float input."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            SENTINEL_IMAGE_SIZE,
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0

    return array


# =========================================================
# LOCATION-BASED SENTINEL-2 ANALYSIS ENDPOINT
# =========================================================

@app.post("/api/analyze-location")
def analyze_location(selection: AnalysisSelectionRequest):
    """Retrieve Sentinel-2 imagery and run an explicitly experimental CNN prediction."""
    if selection.after_date <= selection.before_date:
        raise HTTPException(
            status_code=422,
            detail="after_date must be later than before_date.",
        )

    if selection.before_date > date.today() or selection.after_date > date.today():
        raise HTTPException(
            status_code=422,
            detail="Analysis dates cannot be in the future.",
        )

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with get_database_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_history (
                latitude, longitude, radius_km, before_date, after_date,
                created_at, status, imagery_retrieved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                selection.latitude,
                selection.longitude,
                selection.radius_km,
                selection.before_date.isoformat(),
                selection.after_date.isoformat(),
                created_at,
                "Retrieving Sentinel-2 imagery",
                0,
            ),
        )
        record_id = cursor.lastrowid
        connection.commit()

    imagery_retrieved = False

    try:
        image_paths = retrieve_sentinel2_pair(
            latitude=selection.latitude,
            longitude=selection.longitude,
            radius_km=selection.radius_km,
            before_date=selection.before_date,
            after_date=selection.after_date,
            record_id=record_id,
        )
        imagery_retrieved = True

        before_path = image_paths["before_image"]
        after_path = image_paths["after_image"]
        before_image_array = load_sentinel_image(before_path)
        after_image_array = load_sentinel_image(after_path)

        combined_image = np.concatenate(
            [before_image_array, after_image_array],
            axis=-1,
        )
        model_input = np.expand_dims(combined_image, axis=0)
        model = get_sentinel_model()
        prediction = model.predict(model_input, verbose=0)[0, :, :, 0]

        change_mask = (prediction >= SENTINEL_THRESHOLD).astype(np.uint8)
        changed_pixels = int(np.sum(change_mask))
        total_pixels = int(change_mask.size)
        change_percentage = (
            (changed_pixels / total_pixels) * 100.0 if total_pixels else 0.0
        )

        result_directory = Path(image_paths["analysis_directory"]) / "results"
        result_directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.clip(prediction * 255.0, 0, 255).astype(np.uint8)
        ).save(result_directory / "change_probability.png")
        Image.fromarray(change_mask * 255).save(result_directory / "change_mask.png")

        with get_database_connection() as connection:
            connection.execute(
                """
                UPDATE analysis_history
                SET status = ?, imagery_retrieved = ?
                WHERE id = ?
                """,
                ("Completed - Experimental prediction", 1, record_id),
            )
            connection.commit()

        return {
            "record_id": record_id,
            "status": "completed",
            "validation_status": (
                "Experimental only: this CNN was trained on LEVIR-CD and "
                "has not been validated for Sentinel-2 imagery."
            ),
            "threshold": SENTINEL_THRESHOLD,
            "predicted_change_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": round(change_percentage, 4),
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
            "prediction_mean": float(prediction.mean()),
            "before_image": create_image_data_url(before_image_array),
            "after_image": create_image_data_url(after_image_array),
            "change_map": create_change_map(change_mask),
            "change_overlay": create_change_overlay(after_image_array, change_mask),
            "change_regions": [],
            "imagery_retrieved": True,
            "before_image_path": str(before_path),
            "after_image_path": str(after_path),
        }

    except Exception as error:
        with get_database_connection() as connection:
            connection.execute(
                """
                UPDATE analysis_history
                SET status = ?, imagery_retrieved = ?
                WHERE id = ?
                """,
                (f"Failed: {str(error)[:400]}", int(imagery_retrieved), record_id),
            )
            connection.commit()

        if isinstance(error, (ValueError, FileNotFoundError)):
            raise HTTPException(status_code=400, detail=str(error))

        raise HTTPException(
            status_code=500,
            detail=f"Location-based analysis failed: {error}",
        )


# =========================================================
# ORIGINAL LEVIR-CD FILENAME-BASED ANALYSIS ENDPOINT
# =========================================================

@app.get(
    "/api/analyze/{filename}",
    response_model=AnalysisResult,
)
def analyze_image(filename: str):
    try:
        # Load the original LEVIR-CD image pair.
        image_a, image_b, label = load_image_pair(filename)

        if image_a is None or image_b is None:
            raise ValueError("Satellite image pair could not be loaded.")

        # Run the original CNN prediction and region extraction.
        result = run_change_detection(filename)

        before_image = create_image_data_url(image_a)
        after_image = create_image_data_url(image_b)
        change_map = create_change_map(result["change_map"])
        change_overlay = create_change_overlay(image_b, result["change_map"])

        return {
            "filename": result["filename"],
            "threshold": result["threshold"],
            "predicted_change_pixels": result["predicted_change_pixels"],
            "total_pixels": result["total_pixels"],
            "change_percentage": result["change_percentage"],
            "prediction_min": result["prediction_min"],
            "prediction_max": result["prediction_max"],
            "prediction_mean": result["prediction_mean"],
            "before_image": before_image,
            "after_image": after_image,
            "change_map": change_map,
            "change_overlay": change_overlay,
            "change_regions": result["change_regions"],
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"AI processing failed: {error}",
        )
