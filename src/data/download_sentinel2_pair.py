
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "sentinel2" / "test_shivamogga"

load_dotenv(PROJECT_ROOT / ".env")

CLIENT_ID = os.getenv("COPERNICUS_CLIENT_ID")
CLIENT_SECRET = os.getenv("COPERNICUS_CLIENT_SECRET")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

BBOX = [75.5181, 13.8799, 75.6181, 13.9799]

WIDTH = 512
HEIGHT = 512

# Selected acquisitions from the Catalog API
DATES = {
    "before": "2024-01-21T05:34:59Z",
    "after": "2026-01-20T05:35:07Z",
}


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

def get_access_token():
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "Copernicus credentials are missing from .env"
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )

    response.raise_for_status()

    token = response.json()["access_token"]
    print("Copernicus authentication successful.")

    return token


# --------------------------------------------------
# SATELLITE IMAGE DOWNLOAD
# --------------------------------------------------

def download_image(token, label, acquisition_time):

    print(f"\nDownloading {label} image...")
    print("Acquisition:", acquisition_time)

    payload = {
        "input": {
            "bounds": {
                "bbox": BBOX,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": "2024-01-21T00:00:00Z"
                            if label == "before"
                            else "2026-01-20T00:00:00Z",
                            "to": "2024-01-22T00:00:00Z"
                            if label == "before"
                            else "2026-01-21T00:00:00Z",
                        },
                        "mosaickingOrder": "leastCC",
                    },
                }
            ],
        },
        "output": {
            "width": WIDTH,
            "height": HEIGHT,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/png"
                    },
                }
            ],
        },
        "evalscript": """
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
        """,
    }

    response = requests.post(
        PROCESS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )

    if not response.ok:
        raise RuntimeError(
            f"{label} image download failed: "
            f"{response.status_code} - {response.text[:1000]}"
        )

    image = Image.open(BytesIO(response.content)).convert("RGB")

    output_path = OUTPUT_DIR / f"{label}.png"
    image.save(output_path)

    print("Saved:", output_path)
    print("Dimensions:", image.size)

    return output_path


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    token = get_access_token()

    before_path = download_image(
        token,
        "before",
        DATES["before"],
    )

    after_path = download_image(
        token,
        "after",
        DATES["after"],
    )

    print("\nSatellite image download completed.")
    print("Before:", before_path)
    print("After:", after_path)


if __name__ == "__main__":
    main()
