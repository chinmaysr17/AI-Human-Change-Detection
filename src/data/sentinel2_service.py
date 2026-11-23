
import os
import io
import json
import tarfile
from pathlib import Path
from datetime import datetime, timedelta
from pyproj import Transformer
import rasterio
from rasterio.io import MemoryFile
import requests
import numpy as np
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

MAX_CLOUD_COVERAGE = 10.0


def select_closest_tile(
    tiles,
    target_date,
    max_cloud_coverage=MAX_CLOUD_COVERAGE,
):
    """
    Select the nearest Sentinel-2 acquisition that meets
    the maximum cloud-coverage threshold.
    """

    if not tiles:
        raise RuntimeError(
            "No Sentinel-2 acquisition tiles were returned."
        )

    eligible_tiles = []

    for tile in tiles:
        try:
            date_string = tile.get("date")

            if not date_string:
                continue

            acquisition_date = datetime.fromisoformat(
                date_string.replace("Z", "+00:00")
            )

            comparison_target = target_date

            if comparison_target.tzinfo is None:
                comparison_target = comparison_target.replace(
                    tzinfo=acquisition_date.tzinfo
                )

            elif acquisition_date.tzinfo is None:
                acquisition_date = acquisition_date.replace(
                    tzinfo=comparison_target.tzinfo
                )

            cloud_coverage = float(
                tile.get("cloudCoverage", 100)
            )

            if not 0 <= cloud_coverage <= max_cloud_coverage:
                continue

            date_difference = abs(
                (
                    acquisition_date - comparison_target
                ).total_seconds()
            )

            eligible_tiles.append(
                (date_difference, cloud_coverage, tile)
            )

        except (ValueError, TypeError):
            continue

    if not eligible_tiles:
        raise RuntimeError(
            f"No Sentinel-2 acquisition found with cloud "
            f"coverage at or below {max_cloud_coverage}%."
        )

    eligible_tiles.sort(
        key=lambda item: (item[0], item[1])
    )

    return eligible_tiles[0][2]

# ==================================================
# GEOTIFF VALIDATION
# ==================================================

OUTPUT_CRS = "EPSG:32643"
OUTPUT_WIDTH = 1000
OUTPUT_HEIGHT = 1000
NODATA_VALUE = -9999.0


def validate_geotiff(image_bytes):
    """Validate a georeferenced 3-band Float32 GeoTIFF."""

    try:
        with MemoryFile(image_bytes) as memfile:
            with memfile.open() as dataset:
                if dataset.driver != "GTiff":
                    raise RuntimeError(
                        f"Expected GeoTIFF, received {dataset.driver}."
                    )

                if (dataset.width, dataset.height) != (
                    OUTPUT_WIDTH,
                    OUTPUT_HEIGHT,
                ):
                    raise RuntimeError(
                        "Unexpected GeoTIFF dimensions: "
                        f"{dataset.width} x {dataset.height}."
                    )

                if dataset.count != 3:
                    raise RuntimeError(
                        f"Expected 3 bands, received {dataset.count}."
                    )
                if any(dtype != "float32" for dtype in dataset.dtypes):
                     raise RuntimeError(
                         f"Expected Float32 bands, received {dataset.dtypes}."
                    )

                if dataset.crs is None:
                    raise RuntimeError(
                        "GeoTIFF has no CRS; refusing ungeoreferenced imagery."
                    )

                if dataset.crs.to_string() != OUTPUT_CRS:
                    raise RuntimeError(
                        f"Unexpected CRS: {dataset.crs}; expected {OUTPUT_CRS}."
                    )

                if dataset.nodata is None:
                    raise RuntimeError(
                        "GeoTIFF does not declare a NoData value."
                    )

                if not np.isclose(dataset.nodata, NODATA_VALUE):
                    raise RuntimeError(
                        f"Unexpected NoData value: {dataset.nodata}."
                    )

                data = dataset.read(masked=True)
                valid = np.ma.compressed(data)

                if valid.size == 0:
                    raise RuntimeError("GeoTIFF contains no valid pixels.")

                if not np.isfinite(valid).all():
                    raise RuntimeError(
                        "GeoTIFF contains non-finite valid pixel values."
                    )

                if np.ma.getmaskarray(data).all():
                    raise RuntimeError("All GeoTIFF pixels are NoData.")

                return {
                    "width": dataset.width,
                    "height": dataset.height,
                    "count": dataset.count,
                    "crs": dataset.crs.to_string(),
                    "transform": list(dataset.transform)[:6],
                    "bounds": list(dataset.bounds),
                    "resolution": list(dataset.res),
                    "nodata": float(dataset.nodata),
                    "dtype": dataset.dtypes[0],
                }

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Invalid GeoTIFF returned by Sentinel Hub: {exc}"
        ) from exc


# ==================================================
# PROCESS API TAR RESPONSE EXTRACTION
# ==================================================

def extract_process_response(response_content):
    """Extract default GeoTIFF and userdata.json from TAR response."""

    try:
        with tarfile.open(
            fileobj=io.BytesIO(response_content),
            mode="r:*",
        ) as archive:
            members = archive.getmembers()

            image_member = next(
                (
                    member for member in members
                    if Path(member.name).name.lower()
                    in {"default.tif", "default.tiff"}
                ),
                None,
            )

            if image_member is None:
                image_member = next(
                    (
                        member for member in members
                        if member.name.lower().endswith((".tif", ".tiff"))
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
                    "GeoTIFF was not found in the Process API response."
                )

            if metadata_member is None:
                raise RuntimeError(
                    "userdata.json was not returned by Sentinel Hub."
                )

            image_file = archive.extractfile(image_member)
            metadata_file = archive.extractfile(metadata_member)

            if image_file is None or metadata_file is None:
                raise RuntimeError(
                    "Could not read GeoTIFF or metadata from TAR response."
                )

            image_bytes = image_file.read()
            metadata = json.load(metadata_file)

            return image_bytes, metadata

    except tarfile.TarError as exc:
        raise RuntimeError(
            f"Invalid TAR response from Sentinel Hub: {exc}"
        ) from exc


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

    # Build a square study-area bbox in a metric projected CRS.
    # Shivamogga is in UTM zone 43N (EPSG:32643).
    transformer = Transformer.from_crs(
        "EPSG:4326",
        OUTPUT_CRS,
        always_xy=True,
    )

    center_x, center_y = transformer.transform(
        longitude,
        latitude,
    )

    radius_m = radius_km * 1000.0

    bbox = [
        center_x - radius_m,
        center_y - radius_m,
        center_x + radius_m,
        center_y + radius_m,
    ]

    # Retry with progressively wider acquisition windows.

    search_windows = [15, 30, 45, 60]

    # Inject the requested date into the evalscript.

    target_date_string = target_date.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


    evalscript = f"""
    //VERSION=3

    var TARGET_DATE = "{target_date_string}";
    var MAX_CLOUD_COVERAGE = {MAX_CLOUD_COVERAGE};
    var NODATA = {NODATA_VALUE};

    function setup() {{
        return {{
            input: [{{
                bands: [
                    "B02", "B03", "B04",
                    "SCL", "dataMask"
                ]
            }}],
            mosaicking: Mosaicking.TILE,
            output: {{
                id: "default",
                bands: 3,
                sampleType: "FLOAT32",
                nodataValue: NODATA
            }}
        }};
    }}

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

            var cloudCoverage = Number(tile.cloudCoverage);
            if (
                isNaN(cloudCoverage) ||
                cloudCoverage < 0 ||
                cloudCoverage > MAX_CLOUD_COVERAGE
            ) {{
                continue;
            }}

            var difference = Math.abs(acquisitionTime - targetTime);

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

        if (selectedIndex < 0 || !samples[selectedIndex]) {{
            return [NODATA, NODATA, NODATA];
        }}

        var sample = samples[selectedIndex];

        // SCL: 3 shadow, 8/9 cloud, 10 cirrus, 11 snow/ice.
        // dataMask also removes pixels outside valid source coverage.
        if (
            sample.dataMask !== 1 ||
            sample.SCL === 3 ||
            sample.SCL === 8 ||
            sample.SCL === 9 ||
            sample.SCL === 10 ||
            sample.SCL === 11
        ) {{
            return [NODATA, NODATA, NODATA];
        }}

        // Preserve reflectance values; do not apply display stretching.
        return [sample.B04, sample.B03, sample.B02];
    }}

    function updateOutputMetadata(
        scenes,
        inputMetadata,
        outputMetadata
    ) {{
        var selectedIndex = getSelectedIndex(scenes);

        outputMetadata.userData = {{
            "tiles": scenes.tiles,
            "selectedIndex": selectedIndex,
            "selectedAcquisition": selectedIndex >= 0
                ? scenes.tiles[selectedIndex]
                : null,
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
                            "EPSG/0/32643"
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

                "width": OUTPUT_WIDTH,
                "height": OUTPUT_HEIGHT,

                "responses": [

                    {
                        "identifier": "default",
                        "format": {
                            "type": "image/tiff"
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

            # Verify that the Evalscript rendered the same acquisition.

            rendered_tile = metadata.get("selectedAcquisition")

            if not rendered_tile:
                raise RuntimeError(
                    "Evalscript did not report its selected acquisition."
                )

            if selected_tile.get("date") != rendered_tile.get("date"):
                raise RuntimeError(
                    "Acquisition date mismatch between Python "
                    "and Evalscript."
                )

            for identity_field in (
                "productId",
                "sentinel2ProductId",
                "shId",
            ):
                python_value = selected_tile.get(identity_field)
                rendered_value = rendered_tile.get(identity_field)

                if (
                    python_value is not None
                    and rendered_value is not None
                    and python_value != rendered_value
                ):
                    raise RuntimeError(
                        "Acquisition mismatch for "
                        f"{identity_field}: Python and Evalscript "
                        "selected different Sentinel-2 products."
                    )

            print("Acquisition consistency verified.")

            # Validate geospatial raster before saving.
            raster_info = validate_geotiff(image_bytes)

            # Save the original GeoTIFF bytes without stripping georeferencing.
            output_path = Path(output_path)
            if output_path.suffix.lower() not in {".tif", ".tiff"}:
                output_path = output_path.with_suffix(".tif")

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path.write_bytes(image_bytes)

            # Save metadata.

            metadata_path = output_path.with_name(
                f"{output_path.stem}_metadata.json"
            )

            metadata_record = {

                "requested_date": target_date.isoformat(),

                "selected_acquisition": selected_tile,

                "rendered_acquisition": rendered_tile,

                "max_cloud_coverage_percent": MAX_CLOUD_COVERAGE,

                "search_window_days": window_days,

                "search_start": start_date.isoformat(),

                "search_end": end_date.isoformat(),

                "tiles": tiles,

                "service_version": metadata.get(
                    "serviceVersion"
                ),

                "geospatial": {
                    **raster_info,
                    "requested_crs": OUTPUT_CRS,
                    "requested_bbox": bbox,
                    "requested_width": OUTPUT_WIDTH,
                    "requested_height": OUTPUT_HEIGHT,
                    "band_order": ["B04", "B03", "B02"],
                    "band_description": [
                        "Red", "Green", "Blue"
                    ],
                    "cloud_shadow_mask": True,
                },
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

    before_path = analysis_dir / "before.tif"

    after_path = analysis_dir / "after.tif"

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
