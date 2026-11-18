
import os
from pathlib import Path
from datetime import date, timedelta

import requests
from dotenv import load_dotenv

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/"
    "CDSE/protocol/openid-connect/token"
)

PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

OUTPUT_DIR = BASE_DIR / "data" / "sentinel2" / "analyses"


def get_access_token():
    client_id = os.getenv("COPERNICUS_CLIENT_ID")
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Copernicus credentials are missing. Check your .env file."
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    response.raise_for_status()

    return response.json()["access_token"]


def download_sentinel_image(
    token,
    latitude,
    longitude,
    radius_km,
    target_date,
    output_path,
):
    # Approximate bounding box around the selected location.
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (
        111.0 * max(abs(__import__("math").cos(
            __import__("math").radians(latitude)
        )), 0.01)
    )

    bbox = [
        longitude - lon_offset,
        latitude - lat_offset,
        longitude + lon_offset,
        latitude + lat_offset,
    ]

    start_date = target_date - timedelta(days=15)
    end_date = target_date + timedelta(days=15)

    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"],
                units: "REFLECTANCE"
            }],
            output: {
                bands: 3,
                sampleType: "AUTO"
            }
        };
    }

    function evaluatePixel(sample) {
        return [
            2.5 * sample.B04,
            2.5 * sample.B03,
            2.5 * sample.B02
        ];
    }
    """

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                },
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{start_date.isoformat()}T00:00:00Z",
                        "to": f"{end_date.isoformat()}T23:59:59Z",
                    },
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/png"},
            }],
        },
        "evalscript": evalscript,
    }

    response = requests.post(
        PROCESS_URL,
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=180,
    )
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    return output_path


def retrieve_sentinel2_pair(
    latitude,
    longitude,
    radius_km,
    before_date,
    after_date,
    record_id,
):
    if before_date >= after_date:
        raise ValueError("Before date must be earlier than after date.")

    token = get_access_token()

    analysis_dir = OUTPUT_DIR / f"analysis_{record_id}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    before_path = analysis_dir / "before.png"
    after_path = analysis_dir / "after.png"

    print("Downloading Sentinel-2 before image...")
    download_sentinel_image(
        token,
        latitude,
        longitude,
        radius_km,
        before_date,
        before_path,
    )

    print("Downloading Sentinel-2 after image...")
    download_sentinel_image(
        token,
        latitude,
        longitude,
        radius_km,
        after_date,
        after_path,
    )

    return {
        "before_image": str(before_path),
        "after_image": str(after_path),
        "analysis_directory": str(analysis_dir),
    }
