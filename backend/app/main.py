import base64

import io

import json

import os
import sqlite3
from datetime import date, datetime, timezone



import numpy as np

from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from PIL import Image



from src.inference.change_detector import run_change_detection

from src.data.image_loader import load_image_pair





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



    # Includes geometry and visual features for each region.

    change_regions: list[dict]





# =========================================================

# FASTAPI APPLICATION

# =========================================================



app = FastAPI(

    title="AI Human-Induced Change Detection API",

    version="1.2.0"

)



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],

)

# =========================================================
# LOCAL SQLITE ANALYSIS HISTORY
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIRECTORY, exist_ok=True)
DATABASE_PATH = os.path.join(DATA_DIRECTORY, "analysis_history.db")


def get_database_connection():
    """Create a SQLite connection for the local analysis-history database."""
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_history_database():
    """Create the analysis history table if it does not already exist."""
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

    return {

        "message": (

            "AI Human-Induced Change Detection API is running"

        )

    }





@app.get("/health")

def health_check():

    return {

        "status": "healthy"

    }





# =========================================================

# ANALYSIS LOCATION AND DATE SELECTION ENDPOINT

# =========================================================



@app.post(

    "/api/analysis-selection",

    response_model=AnalysisSelectionResponse

)

def validate_analysis_selection(

    selection: AnalysisSelectionRequest

):

    """

    Validate the selected location and date range.

    This endpoint does not retrieve satellite imagery yet.

    """

    if selection.after_date <= selection.before_date:

        raise HTTPException(

            status_code=422,

            detail="after_date must be later than before_date."

        )



    if selection.before_date > date.today():

        raise HTTPException(

            status_code=422,

            detail="before_date cannot be in the future."

        )



    if selection.after_date > date.today():

        raise HTTPException(

            status_code=422,

            detail="after_date cannot be in the future."

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


# MODEL METADATA

# =========================================================



MODEL_METADATA_PATH = os.path.join(

    os.path.dirname(

        os.path.dirname(

            os.path.dirname(

                os.path.abspath(__file__)

            )

        )

    ),

    "models",

    "change_detection_cnn_final_metadata.json"

)





def load_model_metadata():

    if not os.path.exists(MODEL_METADATA_PATH):

        raise FileNotFoundError(

            f"Model metadata not found: {MODEL_METADATA_PATH}"

        )



    with open(

        MODEL_METADATA_PATH,

        "r",

        encoding="utf-8"

    ) as file:

        return json.load(file)





@app.get("/api/model")

def get_model_metadata():

    try:

        return load_model_metadata()



    except FileNotFoundError as error:

        raise HTTPException(

            status_code=404,

            detail=str(error)

        )



    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(

                "Failed to load model metadata: "

                f"{str(error)}"

            )

        )





# =========================================================

# CONVERT IMAGE TO BASE64 DATA URL

# =========================================================



def create_image_data_url(image_array):

    image_array = np.asarray(image_array)



    if np.issubdtype(

        image_array.dtype,

        np.floating

    ):

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



    image = Image.fromarray(image_array)



    buffer = io.BytesIO()

    image.save(buffer, format="PNG")



    encoded_image = base64.b64encode(

        buffer.getvalue()

    ).decode("utf-8")



    return "data:image/png;base64," + encoded_image





# =========================================================

# CREATE BINARY CHANGE MAP

# =========================================================



def create_change_map(change_mask):

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

    image.save(buffer, format="PNG")



    encoded_image = base64.b64encode(

        buffer.getvalue()

    ).decode("utf-8")



    return "data:image/png;base64," + encoded_image





# =========================================================

# CREATE RED CHANGE OVERLAY

# =========================================================



def create_change_overlay(after_image, change_mask):

    after_array = np.asarray(after_image)



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



    if after_array.ndim == 2:

        after_image_pil = Image.fromarray(

            after_array,

            mode="L"

        ).convert("RGB")



    else:

        after_image_pil = Image.fromarray(

            after_array

        ).convert("RGB")



    mask_image = Image.fromarray(

        np.asarray(

            change_mask,

            dtype=np.uint8

        ) * 255,

        mode="L"

    )



    mask_image = mask_image.resize(

        after_image_pil.size,

        Image.Resampling.NEAREST

    )



    red_layer = Image.new(

        "RGB",

        after_image_pil.size,

        (255, 0, 0)

    )



    highlighted_image = Image.blend(

        after_image_pil,

        red_layer,

        0.45

    )



    overlay = Image.composite(

        highlighted_image,

        after_image_pil,

        mask_image

    )



    buffer = io.BytesIO()

    overlay.save(buffer, format="PNG")



    encoded_image = base64.b64encode(

        buffer.getvalue()

    ).decode("utf-8")



    return "data:image/png;base64," + encoded_image





# =========================================================

# SATELLITE IMAGE ANALYSIS ENDPOINT

# =========================================================



@app.get(

    "/api/analyze/{filename}",

    response_model=AnalysisResult

)

def analyze_image(filename: str):

    try:

        # Load the original satellite image pair.

        image_a, image_b, label = load_image_pair(

            filename

        )



        if image_a is None or image_b is None:

            raise ValueError(

                "Satellite image pair could not be loaded."

            )



        # Run CNN prediction, region extraction,

        # geometric features and visual features.

        result = run_change_detection(filename)



        # Convert images into browser-displayable data URLs.

        before_image = create_image_data_url(image_a)

        after_image = create_image_data_url(image_b)



        change_map = create_change_map(

            result["change_map"]

        )



        change_overlay = create_change_overlay(

            image_b,

            result["change_map"]

        )



        # Return the complete result to the frontend.

        return {

            "filename": result["filename"],

            "threshold": result["threshold"],

            "predicted_change_pixels": (

                result["predicted_change_pixels"]

            ),

            "total_pixels": result["total_pixels"],

            "change_percentage": result["change_percentage"],

            "prediction_min": result["prediction_min"],

            "prediction_max": result["prediction_max"],

            "prediction_mean": result["prediction_mean"],



            "before_image": before_image,

            "after_image": after_image,

            "change_map": change_map,

            "change_overlay": change_overlay,



            # This now includes each region's visual_features.

            "change_regions": result["change_regions"]

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
