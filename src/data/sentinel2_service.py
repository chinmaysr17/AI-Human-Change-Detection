
import os
import io
import json
import math
import tarfile
from pathlib import Path
from datetime import datetime, timedelta

import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/"
    "CDSE/protocol/openid-connect/token"
)

PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

OUTPUT_DIR = BASE_DIR / "data" / "sentinel2" / "analyses"


# ==================================================
# COPERNICUS AUTHENTICATION
# ==================================================

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


# ==================================================
# ACQUISITION SELECTION
# ==================================================

def select_closest_tile(tiles, target_date):
    """
    Select the acquisition closest to the requested date.

    If two acquisitions have the same date distance,
    prefer the one with lower cloud coverage.
    """

    if not tiles:
        raise RuntimeError(
            "No Sentinel-2 acquisition tiles were returned."
        )

    def tile_sort_key(tile):

        date_string = tile.get("date")

        if not date_string:
            return float("inf"), float("inf")

        try:
            acquisition_date = datetime.fromisoformat(
                date_string.replace("Z", "")
            )

            date_difference = abs(
                (acquisition_date - target_date).total_seconds()
            )

            cloud_coverage = float(
                tile.get("cloudCoverage", 100)
            )

            return date_difference, cloud_coverage

        except (ValueError, TypeError):
            return float("inf"), float("inf")

    selected_tile = min(
        tiles,
        key=tile_sort_key,
    )

    if not selected_tile.get("date"):
        raise RuntimeError(
            "Selected acquisition does not contain a valid date."
        )

    return selected_tile


# ==================================================
# IMAGE VALIDATION
# ==================================================

def validate_image(image_bytes):
    """Validate downloaded satellite imagery."""

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image.load()

    except Exception as exc:
        raise RuntimeError(
            f"Invalid image returned by Sentinel Hub: {exc}"
        )

    pixels = np.asarray(image)

    if image.size != (512, 512):
        raise RuntimeError(
            f"Unexpected image dimensions: {image.size}"
        )

    if not np.isfinite(pixels).all():
        raise RuntimeError(
            "Image contains invalid pixel values."
        )

    if pixels.max() == 0:
        raise RuntimeError(
            "Sentinel Hub returned a completely black image."
        )

    unique_colors = np.unique(
        pixels.reshape(-1, 3),
        axis=0,
    ).shape[0]

    if unique_colors < 10:
        raise RuntimeError(
            "Satellite image has insufficient pixel variation."
        )

    valid_pixels = np.count_nonzero(
        np.any(pixels > 0, axis=2)
    )

    if valid_pixels < (
        pixels.shape[0] * pixels.shape[1] * 0.10
    ):
        raise RuntimeError(
            "Most pixels are empty or black. Image rejected."
        )

    return image


# ==================================================
# PROCESS API TAR RESPONSE EXTRACTION
# ==================================================

def extract_process_response(response_content):
    """Extract PNG image and userdata.json from TAR response."""

    try:

        with tarfile.open(
            fileobj=io.BytesIO(response_content),
            mode="r:*",
        ) as archive:

            members = archive.getmembers()

            image_member = next(
                (
                    member for member in members
                    if Path(member.name).name.lower() == "default.png"
                ),
                None,
            )

            if image_member is None:

                image_member = next(
                    (
                        member for member in members
                        if member.name.lower().endswith(".png")
                        and "userdata" not in member.name.lower()
                    ),
                    None,
                )

            metadata_member = next(
                (
                    member for member in members
                    if Path(member.name).name.lower() == "userdata.json"
                ),
                None,
            )

            if image_member is None:
                raise RuntimeError(
                    "PNG image was not found in the Process API response."
                )

            if metadata_member is None:
                raise RuntimeError(
                    "userdata.json was not returned by Sentinel Hub."
                )

            image_file = archive.extractfile(image_member)
            metadata_file = archive.extractfile(metadata_member)

            if image_file is None or metadata_file is None:
                raise RuntimeError(
                    "Could not read image or metadata from TAR response."
                )

            image_bytes = image_file.read()
            metadata = json.load(metadata_file)

            return image_bytes, metadata

    except tarfile.TarError as exc:
        raise RuntimeError(
            f"Invalid TAR response from Sentinel Hub: {exc}"
        )


# ==================================================
# SENTINEL-2 IMAGE DOWNLOAD
# ==================================================

def download_sentinel_image(
    token,
    latitude,
    longitude,
    radius_km,
    target_date,
    output_path,
):

    if not isinstance(target_date, datetime):
        raise TypeError(
            "target_date must be a Python datetime object."
        )

    # Calculate bounding box.

    lat_offset = radius_km / 111.0

    lon_offset = radius_km / (
        111.0 * max(
            abs(math.cos(math.radians(latitude))),
            0.01,
        )
    )

    bbox = [
        longitude - lon_offset,
        latitude - lat_offset,
        longitude + lon_offset,
        latitude + lat_offset,
    ]

    # Retry with progressively wider acquisition windows.

    search_windows = [15, 30, 45, 60]

    # Inject the requested date into the evalscript.

    target_date_string = target_date.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    evalscript = f"""
    //VERSION=3

    var TARGET_DATE = "{target_date_string}";

    function setup() {{

        return {{

            input: [{{
                bands: ["B02", "B03", "B04", "dataMask"]
            }}],

            mosaicking: Mosaicking.TILE,

            output: {{
                id: "default",
                bands: 3,
                sampleType: "AUTO"
            }}
        }};
    }}

    // Select ONE acquisition closest to the requested date.

    function getSelectedIndex(scenes) {{

        var targetTime = Date.parse(TARGET_DATE);

        var selectedIndex = -1;
        var smallestDifference = Infinity;
        var lowestCloudCoverage = Infinity;

        for (var i = 0; i < scenes.tiles.length; i++) {{

            var tile = scenes.tiles[i];

            if (!tile.date) {{
                continue;
            }}

            var acquisitionTime = Date.parse(tile.date);

            if (isNaN(acquisitionTime)) {{
                continue;
            }}

            var difference = Math.abs(
                acquisitionTime - targetTime
            );

            var cloudCoverage = Number(
                tile.cloudCoverage
            );

            if (isNaN(cloudCoverage)) {{
                cloudCoverage = 100;
            }}

            if (
                difference < smallestDifference ||
                (
                    difference === smallestDifference &&
                    cloudCoverage < lowestCloudCoverage
                )
            ) {{

                smallestDifference = difference;
                lowestCloudCoverage = cloudCoverage;
                selectedIndex = i;
            }}
        }}

        return selectedIndex;
    }}

    function evaluatePixel(samples, scenes) {{

        var selectedIndex = getSelectedIndex(scenes);

        if (selectedIndex < 0) {{
            return [0, 0, 0];
        }}

        var selectedSample = samples[selectedIndex];

        // Never mix pixels from different acquisition dates.

        if (selectedSample.dataMask !== 1) {{
            return [0, 0, 0];
        }}

        return [
            2.5 * selectedSample.B04,
            2.5 * selectedSample.B03,
            2.5 * selectedSample.B02
        ];
    }}

    function updateOutputMetadata(
        scenes,
        inputMetadata,
        outputMetadata
    ) {{

        outputMetadata.userData = {{
            "tiles": scenes.tiles,
            "serviceVersion": inputMetadata.serviceVersion,
            "requestedDate": TARGET_DATE
        }};
    }}
    """

    last_error = None

    # Search around the requested date.

    for window_days in search_windows:

        start_date = target_date - timedelta(
            days=window_days
        )

        end_date = target_date + timedelta(
            days=window_days
        )

        # Correct ISO date formatting.

        start_datetime = start_date.strftime(
            "%Y-%m-%dT00:00:00Z"
        )

        end_datetime = end_date.strftime(
            "%Y-%m-%dT23:59:59Z"
        )

        payload = {

            "input": {

                "bounds": {

                    "bbox": bbox,

                    "properties": {
                        "crs": (
                            "http://www.opengis.net/def/crs/"
                            "EPSG/0/4326"
                        )
                    },
                },

                "data": [{

                    "type": "sentinel-2-l2a",

                    "dataFilter": {

                        "timeRange": {
                            "from": start_datetime,
                            "to": end_datetime,
                        },

                        "mosaickingOrder": "leastCC",
                    },
                }],
            },

            "output": {

                "width": 512,
                "height": 512,

                "responses": [

                    {
                        "identifier": "default",
                        "format": {
                            "type": "image/png"
                        },
                    },

                    {
                        "identifier": "userdata",
                        "format": {
                            "type": "application/json"
                        },
                    },
                ],
            },

            "evalscript": evalscript,
        }

        try:

            print(
                f"Searching imagery around {target_date.date()} "
                f"(+/- {window_days} days)..."
            )

            response = requests.post(
                PROCESS_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/x-tar",
                },
                timeout=180,
            )

            # Display actual API error details.

            if not response.ok:

                raise RuntimeError(
                    f"Sentinel Hub HTTP {response.status_code}: "
                    f"{response.text[:2000]}"
                )

            # Extract image and metadata.

            image_bytes, metadata = extract_process_response(
                response.content
            )

            tiles = metadata.get("tiles", [])

            if not tiles:
                raise RuntimeError(
                    "No acquisition tile metadata was returned."
                )

            # Determine selected acquisition.

            selected_tile = select_closest_tile(
                tiles,
                target_date,
            )

            # Validate downloaded image.

            image = validate_image(image_bytes)

            # Save image.

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            image.save(
                output_path,
                format="PNG",
            )

            # Save metadata.

            metadata_path = output_path.with_name(
                f"{output_path.stem}_metadata.json"
            )

            metadata_record = {

                "requested_date": target_date.isoformat(),

                "selected_acquisition": selected_tile,

                "search_window_days": window_days,

                "search_start": start_date.isoformat(),

                "search_end": end_date.isoformat(),

                "tiles": tiles,

                "service_version": metadata.get(
                    "serviceVersion"
                ),
            }

            with open(
                metadata_path,
                "w",
                encoding="utf-8",
            ) as metadata_file:

                json.dump(
                    metadata_record,
                    metadata_file,
                    indent=4,
                )

            # Display selected acquisition details.

            print("\nSelected Sentinel-2 acquisition:")

            print(
                "Requested date:",
                target_date.date(),
            )

            print(
                "Actual acquisition:",
                selected_tile.get("date"),
            )

            print(
                "Cloud coverage:",
                selected_tile.get("cloudCoverage"),
                "%",
            )

            print(
                "Product ID:",
                selected_tile.get("productId"),
            )

            print(
                "\nSatellite image saved:",
                output_path,
            )

            print(
                "Metadata saved:",
                metadata_path,
            )

            return output_path

        except Exception as exc:

            last_error = exc

            print(
                f"Image attempt rejected: {exc}"
            )

    raise RuntimeError(
        f"Could not retrieve valid imagery for {target_date}. "
        f"Last error: {last_error}"
    )


# ==================================================
# RETRIEVE BEFORE AND AFTER SENTINEL-2 IMAGES
# ==================================================

def retrieve_sentinel2_pair(
    latitude,
    longitude,
    radius_km,
    before_date,
    after_date,
    record_id,
):

    if not isinstance(before_date, datetime):
        raise TypeError(
            "before_date must be a datetime object."
        )

    if not isinstance(after_date, datetime):
        raise TypeError(
            "after_date must be a datetime object."
        )

    if before_date >= after_date:

        raise ValueError(
            "Before date must be earlier than after date."
        )

    token = get_access_token()

    analysis_dir = OUTPUT_DIR / f"analysis_{record_id}"

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    before_path = analysis_dir / "before.png"

    after_path = analysis_dir / "after.png"

    print(
        "\nDownloading BEFORE Sentinel-2 image..."
    )

    download_sentinel_image(
        token,
        latitude,
        longitude,
        radius_km,
        before_date,
        before_path,
    )

    print(
        "\nDownloading AFTER Sentinel-2 image..."
    )

    download_sentinel_image(
        token,
        latitude,
        longitude,
        radius_km,
        after_date,
        after_path,
    )

    print(
        "\nBoth Sentinel-2 images downloaded successfully."
    )

    return {

        "before_image": str(before_path),

        "after_image": str(after_path),

        "before_metadata": str(
            before_path.with_name("before_metadata.json")
        ),

        "after_metadata": str(
            after_path.with_name("after_metadata.json")
        ),

        "analysis_directory": str(analysis_dir),
    }
