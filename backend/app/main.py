import base64
import io
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from tensorflow.keras.models import load_model

from backend.app.auth_router import (
    get_current_user,
    router as auth_router,
)
from backend.app.database import get_db
from backend.app.models import AnalysisHistory
from src.data.image_loader import load_image_pair
from src.preprocessing.sentinel2_preprocessor import load_sentinel_pair
from src.data.sentinel2_service import (
    retrieve_sentinel2_pair,
    check_sentinel2_pair_availability,
)
from src.inference.change_detector import run_change_detection


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

FRONTEND_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-CSRF-Token"],
)

# =========================================================
# AUTHENTICATION ROUTER
# =========================================================

app.include_router(auth_router)


# =========================================================
# PROJECT PATHS AND POSTGRESQL
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIRECTORY = PROJECT_ROOT / "data"
DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

SENTINEL_MODEL_PATH = (
    PROJECT_ROOT / "models" / "change_detection_cnn_final.keras"
)

SENTINEL_IMAGE_SIZE = (256, 256)
SENTINEL_THRESHOLD = 0.50

_sentinel_model = None


# =========================================================
# BASIC ENDPOINTS
# =========================================================

@app.get("/")
def root():
    return {"message": "AI Human-Induced Change Detection API is running"}


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Report API readiness only when the PostgreSQL connection is available."""
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Database connection is unavailable.",
        )

    return {
        "status": "healthy",
        "database": "connected",
    }


# =========================================================
# LOCATION AND DATE SELECTION ENDPOINT
# =========================================================

@app.post(
    "/api/analysis-selection",
    response_model=AnalysisSelectionResponse,
)
def validate_analysis_selection(
    selection: AnalysisSelectionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
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

    record = AnalysisHistory(
        latitude=selection.latitude,
        longitude=selection.longitude,
        radius_km=selection.radius_km,
        before_date=selection.before_date.isoformat(),
        after_date=selection.after_date.isoformat(),
        created_at=created_at,
        status="Pending Analysis",
        imagery_retrieved=0,
        user_id=current_user.id,
    )

    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to save the analysis selection.",
        )

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
        "record_id": record.id,
        "created_at": created_at,
        "analysis_status": "Pending Analysis",
    }


# =========================================================
# ANALYSIS HISTORY ENDPOINT
# =========================================================

# =========================================================
# SENTINEL-2 HISTORICAL IMAGERY AVAILABILITY ENDPOINT
# =========================================================

@app.post("/api/check-availability")
def check_imagery_availability(
    selection: AnalysisSelectionRequest,
    current_user=Depends(get_current_user),
):
    """Check historical Sentinel-2 availability without downloading imagery."""

    if selection.after_date <= selection.before_date:
        raise HTTPException(
            status_code=422,
            detail="after_date must be later than before_date.",
        )

    if (
        selection.before_date > date.today()
        or selection.after_date > date.today()
    ):
        raise HTTPException(
            status_code=422,
            detail="Availability dates cannot be in the future.",
        )

    try:
        availability = check_sentinel2_pair_availability(
            latitude=selection.latitude,
            longitude=selection.longitude,
            radius_km=selection.radius_km,
            before_date=datetime.combine(
                selection.before_date,
                datetime.min.time(),
            ),
            after_date=datetime.combine(
                selection.after_date,
                datetime.min.time(),
            ),
        )

        return {
            "status": (
                "available"
                if availability["available"]
                else "unavailable"
            ),
            "message": availability["message"],
            "latitude": selection.latitude,
            "longitude": selection.longitude,
            "radius_km": selection.radius_km,
            "before_date": selection.before_date.isoformat(),
            "after_date": selection.after_date.isoformat(),
            "availability": availability,
        }

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Sentinel-2 availability check failed: {str(error)}",
        ) from error

@app.get("/api/analysis-history")
def get_analysis_history(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Return the logged-in user's saved analysis selections, newest first."""

    try:
        rows = db.scalars(
            select(AnalysisHistory)
            .where(AnalysisHistory.user_id == current_user.id)
            .order_by(AnalysisHistory.id.desc())
        ).all()

        records = [
            {
                "id": row.id,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "radius_km": row.radius_km,
                "before_date": row.before_date,
                "after_date": row.after_date,
                "created_at": row.created_at,
                "status": row.status,
                "imagery_retrieved": bool(row.imagery_retrieved),
            }
            for row in rows
        ]

        return {
            "count": len(records),
            "records": records,
        }

    except SQLAlchemyError:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analysis history.",
        )


# =========================================================
# AUTHENTICATED DASHBOARD STATISTICS
# =========================================================

@app.get("/api/dashboard-stats")
def get_dashboard_statistics(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Return dashboard metrics scoped to the authenticated user."""

    try:
        user_filter = AnalysisHistory.user_id == current_user.id

        total_analyses = db.scalar(
            select(func.count(AnalysisHistory.id)).where(user_filter)
        ) or 0

        completed_filter = (
            user_filter
            & AnalysisHistory.status.like("Completed%")
        )

        completed_analyses = db.scalar(
            select(func.count(AnalysisHistory.id)).where(completed_filter)
        ) or 0

        completed_locations = db.execute(
            select(
                AnalysisHistory.latitude,
                AnalysisHistory.longitude,
            )
            .where(completed_filter)
            .distinct()
        ).all()

        # Round coordinates to five decimal places to avoid treating
        # insignificant floating-point differences as separate locations.
        unique_locations = {
            (round(latitude, 5), round(longitude, 5))
            for latitude, longitude in completed_locations
        }

        try:
            metadata = load_model_metadata()
            evaluation = metadata.get("evaluation", {})
            dataset = metadata.get("dataset", {})
            model_f1_percent = evaluation.get("test_f1_percent")
            model_f1_dataset = dataset.get("name")
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            model_f1_percent = None
            model_f1_dataset = None

        return {
            "total_analyses": int(total_analyses),
            "completed_analyses": int(completed_analyses),
            "unique_locations": len(unique_locations),
            "model_f1_percent": model_f1_percent,
            "model_f1_dataset": model_f1_dataset,
            "model_f1_context": (
                "LEVIR-CD test-set metric; not validated on Sentinel-2."
                if model_f1_percent is not None
                else "Model evaluation metric unavailable."
            ),
        }

    except SQLAlchemyError:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve dashboard statistics.",
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
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to load model metadata.",
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
            raise FileNotFoundError(
                f"Trained CNN model not found: {SENTINEL_MODEL_PATH}"
            )
        _sentinel_model = load_model(SENTINEL_MODEL_PATH, compile=False)

    return _sentinel_model


# =========================================================
# LOCATION-BASED SENTINEL-2 ANALYSIS ENDPOINT
# =========================================================

@app.post("/api/analyze-location")
def analyze_location(
    selection: AnalysisSelectionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
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

    record = AnalysisHistory(
        latitude=selection.latitude,
        longitude=selection.longitude,
        radius_km=selection.radius_km,
        before_date=selection.before_date.isoformat(),
        after_date=selection.after_date.isoformat(),
        created_at=created_at,
        status="Retrieving Sentinel-2 imagery",
        imagery_retrieved=0,
        user_id=current_user.id,
    )

    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to create the analysis history record.",
        )

    record_id = record.id
    imagery_retrieved = False

    try:
        image_pair = retrieve_sentinel2_pair(
            latitude=selection.latitude,
            longitude=selection.longitude,
            radius_km=selection.radius_km,
            before_date=datetime.combine(
                selection.before_date,
                datetime.min.time(),
            ),
            after_date=datetime.combine(
                selection.after_date,
                datetime.min.time(),
            ),
        )
        imagery_retrieved = True

        (
            model_input,
            before_image_array,
            after_image_array,
            valid_mask,
        ) = load_sentinel_pair(
            image_pair["before_image"],
            image_pair["after_image"],
        )

        valid_pixel_count = int(np.sum(valid_mask))
        if valid_pixel_count == 0:
            raise ValueError("No valid overlapping pixels are available for analysis.")

        model = get_sentinel_model()
        prediction = model.predict(model_input, verbose=0)[0, :, :, 0]

        if prediction.shape != valid_mask.shape:
            raise ValueError("CNN prediction dimensions do not match the validity mask.")

        valid_predictions = prediction[valid_mask]
        change_mask = (
            (prediction >= SENTINEL_THRESHOLD) & valid_mask
        ).astype(np.uint8)

        changed_pixels = int(np.sum(change_mask))
        total_pixels = valid_pixel_count
        change_percentage = (changed_pixels / total_pixels) * 100.0

        record_to_update = db.scalar(
            select(AnalysisHistory).where(
                AnalysisHistory.id == record_id,
                AnalysisHistory.user_id == current_user.id,
            )
        )
        if record_to_update is None:
            raise RuntimeError("Analysis history record could not be found.")

        record_to_update.status = "Completed - Experimental prediction"
        record_to_update.imagery_retrieved = 1
        db.commit()

        response_payload = {
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
            "prediction_min": float(valid_predictions.min()),
            "prediction_max": float(valid_predictions.max()),
            "prediction_mean": float(valid_predictions.mean()),
            "before_image": create_image_data_url(before_image_array),
            "after_image": create_image_data_url(after_image_array),
            "change_map": create_change_map(change_mask),
            "change_overlay": create_change_overlay(after_image_array, change_mask),
            "change_regions": [],
            "imagery_retrieved": True,
        }

        # Drop source GeoTIFF buffers and intermediate arrays after creating
        # the response previews. Only metadata/history and preview strings remain.
        del image_pair, model_input, prediction, valid_predictions
        del before_image_array, after_image_array, valid_mask, change_mask

        return response_payload

    except Exception as error:
        db.rollback()

        try:
            record_to_update = db.scalar(
                select(AnalysisHistory).where(
                    AnalysisHistory.id == record_id,
                    AnalysisHistory.user_id == current_user.id,
                )
            )
            if record_to_update is not None:
                record_to_update.status = f"Failed: {str(error)[:400]}"
                record_to_update.imagery_retrieved = int(imagery_retrieved)
                db.commit()
        except SQLAlchemyError:
            db.rollback()

        if isinstance(error, (ValueError, FileNotFoundError)):
            raise HTTPException(status_code=400, detail=str(error))

        if isinstance(error, HTTPException):
            raise error

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
