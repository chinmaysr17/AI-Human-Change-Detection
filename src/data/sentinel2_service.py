
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
CATALOG_URL = "https://sh.dataspace.copernicus.eu/catalog/v1/search"
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
# GLOBAL GEOSPATIAL CONFIGURATION
# ==================================================

OUTPUT_WIDTH = 1000
OUTPUT_HEIGHT = 1000
NODATA_VALUE = -9999.0


def get_local_utm_crs(latitude, longitude):
    """
    Select a suitable projected CRS for a geographic location.

    Uses UTM for latitudes between 80°S and 84°N.
    Uses polar stereographic projections outside that range.
    """

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    # Arctic and Antarctic polar regions
    if latitude >= 84:
        return "EPSG:3413"

    if latitude <= -80:
        return "EPSG:3031"

    # Calculate the appropriate UTM zone
    zone = int((longitude + 180) // 6) + 1
    zone = max(1, min(zone, 60))

    if latitude >= 0:
        epsg_code = 32600 + zone
    else:
        epsg_code = 32700 + zone

    return f"EPSG:{epsg_code}"


def validate_geotiff(image_bytes, expected_crs):
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

                if dataset.crs != rasterio.crs.CRS.from_user_input(expected_crs):
                    raise RuntimeError(
        f"Unexpected CRS: {dataset.crs}; expected {expected_crs}."
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


def check_sentinel2_availability(
    token,
    latitude,
    longitude,
    radius_km,
    target_date,
    output_crs=None,
    max_cloud_coverage=MAX_CLOUD_COVERAGE,
):
    """
    Check Sentinel-2 L2A catalogue availability without
    downloading or processing satellite imagery.
    """

    if not isinstance(target_date, datetime):
        raise TypeError("target_date must be a datetime object.")

    if radius_km <= 0:
        raise ValueError("radius_km must be greater than zero.")

    if output_crs is None:
        output_crs = get_local_utm_crs(latitude, longitude)

    # Create the same projected bounding box used by image downloads.
    transformer = Transformer.from_crs(
        "EPSG:4326",
        output_crs,
        always_xy=True,
    )

    center_x, center_y = transformer.transform(
        longitude,
        latitude,
    )

    radius_m = radius_km * 1000.0

    bbox_utm = [
        center_x - radius_m,
        center_y - radius_m,
        center_x + radius_m,
        center_y + radius_m,
    ]

    # Catalog API requires WGS84 longitude/latitude coordinates.
    to_wgs84 = Transformer.from_crs(
        output_crs,
        "EPSG:4326",
        always_xy=True,
    )

    corners = [
        to_wgs84.transform(bbox_utm[0], bbox_utm[1]),
        to_wgs84.transform(bbox_utm[0], bbox_utm[3]),
        to_wgs84.transform(bbox_utm[2], bbox_utm[1]),
        to_wgs84.transform(bbox_utm[2], bbox_utm[3]),
    ]

    bbox_wgs84 = [
        min(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[0] for point in corners),
        max(point[1] for point in corners),
    ]

    search_windows = [15, 30, 45, 60]

    for window_days in search_windows:

        start_date = target_date - timedelta(days=window_days)
        end_date = target_date + timedelta(days=window_days)

        start_datetime = start_date.strftime(
            "%Y-%m-%dT00:00:00Z"
        )
        end_datetime = end_date.strftime(
            "%Y-%m-%dT23:59:59Z"
        )

        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox_wgs84,
            "datetime": f"{start_datetime}/{end_datetime}",
            "limit": 100,
        }

        print(
            f"Checking Sentinel-2 catalogue around "
            f"{target_date.date()} (+/- {window_days} days)..."
        )

        response = requests.post(
            CATALOG_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )

        if not response.ok:
            raise RuntimeError(
                f"Sentinel Hub Catalog API HTTP "
                f"{response.status_code}: {response.text[:2000]}"
            )

        catalog_data = response.json()
        features = catalog_data.get("features", [])

        # Convert Catalog API metadata into the format
        # expected by the existing acquisition selector.
        tiles = []

        for feature in features:

            properties = feature.get("properties", {})

            acquisition_date = properties.get("datetime")
            cloud_coverage = properties.get("eo:cloud_cover")

            if not acquisition_date or cloud_coverage is None:
                continue

            tiles.append({
                "date": acquisition_date,
                "cloudCoverage": cloud_coverage,
                "productId": feature.get("id"),
                "catalogId": feature.get("id"),
                "collection": feature.get("collection"),
            })

        if not tiles:
            print("No catalogue acquisitions found in this window.")
            continue

        try:
            selected_tile = select_closest_tile(
                tiles,
                target_date,
                max_cloud_coverage=max_cloud_coverage,
            )
        except RuntimeError:
            print(
                f"No acquisition satisfies the "
                f"{max_cloud_coverage}% cloud limit."
            )
            continue

        return {
            "available": True,
            "requested_date": target_date.isoformat(),
            "selected_acquisition": selected_tile,
            "actual_acquisition_date": selected_tile["date"],
            "cloud_coverage_percent": float(
                selected_tile["cloudCoverage"]
            ),
            "product_id": selected_tile["productId"],
            "search_window_days": window_days,
            "search_start": start_date.isoformat(),
            "search_end": end_date.isoformat(),
            "catalog_results_count": len(features),
            "eligible_results_count": len(tiles),
            "max_cloud_coverage_percent": max_cloud_coverage,
            "catalog_bbox_wgs84": bbox_wgs84,
        }

    return {
        "available": False,
        "requested_date": target_date.isoformat(),
        "selected_acquisition": None,
        "actual_acquisition_date": None,
        "cloud_coverage_percent": None,
        "product_id": None,
        "max_cloud_coverage_percent": max_cloud_coverage,
        "message": (
            "No suitable Sentinel-2 acquisition was found "
            "within the 60-day search window."
        ),
    }

def check_sentinel2_pair_availability(
    latitude,
    longitude,
    radius_km,
    before_date,
    after_date,
    max_cloud_coverage=MAX_CLOUD_COVERAGE,
):
    """Check availability of both images without downloading them."""

    if not isinstance(before_date, datetime):
        raise TypeError("before_date must be a datetime object.")

    if not isinstance(after_date, datetime):
        raise TypeError("after_date must be a datetime object.")

    if before_date >= after_date:
        raise ValueError("Before date must be earlier than after date.")

    token = get_access_token()
    output_crs = get_local_utm_crs(latitude, longitude)

    before_result = check_sentinel2_availability(
        token=token,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        target_date=before_date,
        output_crs=output_crs,
        max_cloud_coverage=max_cloud_coverage,
    )

    after_result = check_sentinel2_availability(
        token=token,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        target_date=after_date,
        output_crs=output_crs,
        max_cloud_coverage=max_cloud_coverage,
    )

    pair_available = (
        before_result["available"]
        and after_result["available"]
    )

    return {
        "available": pair_available,
        "before": before_result,
        "after": after_result,
        "message": (
            "Both Sentinel-2 acquisitions are available."
            if pair_available
            else "A suitable acquisition was not found for both dates."
        ),
    }

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
    output_crs,
):

    if not isinstance(target_date, datetime):
        raise TypeError(
            "target_date must be a Python datetime object."
        )

    # Build a square study-area bbox using the selected local CRS.
    transformer = Transformer.from_crs(
    "EPSG:4326",
    output_crs,
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
                            f"EPSG/0/{output_crs.split(':')[1]}"
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
            raster_info = validate_geotiff(image_bytes, output_crs)

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
                    "requested_crs":  output_crs,
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
    output_crs = get_local_utm_crs(latitude, longitude)

    print(f"Selected CRS for location: {output_crs}")
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
        output_crs=output_crs,
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
        output_crs=output_crs,
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
