"""
Sentinel-2 SCL diagnostic for the Shivamogga before/after pair.

Retrieves B04/B03/B02 plus the raw SCL classification as a 4th Float32 band.
It intentionally does NOT apply cloud masking: the purpose is to measure the
original SCL class distribution for each selected acquisition.

Run from project root:
python src/data/sentinel2_scl_diagnostic.py
"""

import io
import json
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import requests
from pyproj import Transformer

from src.data.sentinel2_service import (
    BASE_DIR,
    PROCESS_URL,
    MAX_CLOUD_COVERAGE,
    get_access_token,
    select_closest_tile,
)

LATITUDE = 13.9299
LONGITUDE = 75.5681
RADIUS_KM = 5
OUTPUT_DIR = BASE_DIR / "data" / "sentinel2" / "scl_diagnostic"
DATES = {
    "before": datetime(2016, 1, 20),
    "after": datetime(2026, 1, 20),
}
TARGET_CRS = "EPSG:32643"
TARGET_CRS_URI = "http://www.opengis.net/def/crs/EPSG/0/32643"
NODATA = -9999.0
WIDTH = 1000
HEIGHT = 1000

# SCL class names from Sentinel-2 L2A Scene Classification.
SCL_NAMES = {
    0: "No data",
    1: "Saturated or defective",
    2: "Dark area pixels",
    3: "Cloud shadows",
    4: "Vegetation",
    5: "Bare soil",
    6: "Water",
    7: "Unclassified",
    8: "Cloud medium probability",
    9: "Cloud high probability",
    10: "Thin cirrus",
    11: "Snow or ice",
}


def make_bbox():
    transformer = Transformer.from_crs(
        "EPSG:4326", TARGET_CRS, always_xy=True
    )
    center_x, center_y = transformer.transform(LONGITUDE, LATITUDE)
    offset = RADIUS_KM * 1000
    return [
        center_x - offset,
        center_y - offset,
        center_x + offset,
        center_y + offset,
    ]


def build_evalscript(target_date):
    target_date_string = target_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""//VERSION=3
var TARGET_DATE = "{target_date_string}";
var MAX_CLOUD_COVERAGE = {MAX_CLOUD_COVERAGE};

function setup() {{
    return {{
        input: [{{ bands: ["B02", "B03", "B04", "SCL", "dataMask"] }}],
        mosaicking: Mosaicking.TILE,
        output: {{
            id: "default",
            bands: 4,
            sampleType: "FLOAT32",
            nodataValue: {NODATA}
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
        if (!tile.date) continue;

        var acquisitionTime = Date.parse(tile.date);
        if (isNaN(acquisitionTime)) continue;

        var cloudCoverage = Number(tile.cloudCoverage);
        if (isNaN(cloudCoverage) || cloudCoverage < 0 ||
            cloudCoverage > MAX_CLOUD_COVERAGE) continue;

        var difference = Math.abs(acquisitionTime - targetTime);
        if (difference < smallestDifference ||
            (difference === smallestDifference &&
             cloudCoverage < lowestCloudCoverage)) {{
            smallestDifference = difference;
            lowestCloudCoverage = cloudCoverage;
            selectedIndex = i;
        }}
    }}
    return selectedIndex;
}}

function evaluatePixel(samples, scenes) {{
    var index = getSelectedIndex(scenes);
    if (index < 0 || !samples[index]) {{
        return [{NODATA}, {NODATA}, {NODATA}, {NODATA}];
    }}
    var s = samples[index];
    // Keep raw SCL values even for cloud/shadow pixels so their
    // distribution can be measured. dataMask invalid pixels are NoData.
    if (s.dataMask !== 1) {{
        return [{NODATA}, {NODATA}, {NODATA}, {NODATA}];
    }}
    return [s.B04, s.B03, s.B02, s.SCL];
}}

function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {{
    var index = getSelectedIndex(scenes);
    outputMetadata.userData = {{
        "tiles": scenes.tiles,
        "selectedIndex": index,
        "selectedAcquisition": index >= 0 ? scenes.tiles[index] : null,
        "serviceVersion": inputMetadata.serviceVersion,
        "requestedDate": TARGET_DATE
    }};
}}
"""


def extract_tar(content):
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as archive:
        members = archive.getmembers()
        raster_member = next(
            (m for m in members if Path(m.name).name.lower() in
             ("default.tif", "default.tiff")),
            None,
        )
        if raster_member is None:
            raster_member = next(
                (m for m in members if m.name.lower().endswith((".tif", ".tiff"))
                 and "userdata" not in m.name.lower()),
                None,
            )
        metadata_member = next(
            (m for m in members if Path(m.name).name.lower() == "userdata.json"),
            None,
        )
        if raster_member is None or metadata_member is None:
            raise RuntimeError(
                "Process API TAR must contain a TIFF and userdata.json. "
                f"Found: {[m.name for m in members]}"
            )
        raster_file = archive.extractfile(raster_member)
        metadata_file = archive.extractfile(metadata_member)
        if raster_file is None or metadata_file is None:
            raise RuntimeError("Could not read TIFF or userdata.json from TAR.")
        return raster_file.read(), json.load(metadata_file)


def retrieve_one(token, label, target_date, bbox):
    start = target_date - timedelta(days=15)
    end = target_date + timedelta(days=15)
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": TARGET_CRS_URI},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": start.strftime("%Y-%m-%dT00:00:00Z"),
                        "to": end.strftime("%Y-%m-%dT23:59:59Z"),
                    },
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width": WIDTH,
            "height": HEIGHT,
            "responses": [
                {"identifier": "default", "format": {"type": "image/tiff"}},
                {"identifier": "userdata", "format": {"type": "application/json"}},
            ],
        },
        "evalscript": build_evalscript(target_date),
    }

    print(f"\nRetrieving {label.upper()} SCL diagnostic for {target_date.date()}...")
    response = requests.post(
        PROCESS_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/x-tar",
        },
        timeout=180,
    )
    if not response.ok:
        raise RuntimeError(
            f"Sentinel Hub HTTP {response.status_code}: {response.text[:1500]}"
        )

    tiff_bytes, metadata = extract_tar(response.content)
    tiles = metadata.get("tiles", [])
    selected = select_closest_tile(tiles, target_date)
    rendered = metadata.get("selectedAcquisition")
    if not rendered:
        raise RuntimeError("Evalscript did not return selectedAcquisition metadata.")
    if selected.get("date") != rendered.get("date"):
        raise RuntimeError("Python and Evalscript selected different acquisition dates.")
    for field in ("productId", "sentinel2ProductId", "shId"):
        if selected.get(field) is not None and rendered.get(field) is not None:
            if selected[field] != rendered[field]:
                raise RuntimeError(f"Acquisition mismatch for {field}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{label}_scl_diagnostic.tif"
    output_path.write_bytes(tiff_bytes)
    metadata_path = OUTPUT_DIR / f"{label}_scl_diagnostic_metadata.json"
    metadata_path.write_text(
        json.dumps({
            "requested_date": target_date.isoformat(),
            "selected_acquisition": selected,
            "rendered_acquisition": rendered,
            "tiles": tiles,
            "service_version": metadata.get("serviceVersion"),
            "diagnostic_bands": ["B04", "B03", "B02", "SCL"],
        }, indent=2),
        encoding="utf-8",
    )
    print("Selected:", selected.get("date"))
    print("Cloud coverage:", selected.get("cloudCoverage"), "%")
    print("Saved:", output_path)
    return output_path


def report_scl(path, label):
    with rasterio.open(path) as ds:
        if ds.count != 4:
            raise RuntimeError(f"{label}: expected 4 bands, found {ds.count}")
        scl = ds.read(4)
        valid = np.isfinite(scl) & (scl != NODATA)
        values = np.rint(scl[valid]).astype(np.int16)
        print(f"\n===== {label.upper()} SCL DISTRIBUTION =====")
        print("CRS:", ds.crs, "| Size:", ds.width, "x", ds.height)
        print("SCL-valid pixels:", f"{valid.mean() * 100:.2f}%")
        total = max(values.size, 1)
        counts = np.unique(values, return_counts=True)
        count_map = dict(zip(counts[0].tolist(), counts[1].tolist()))
        for code in range(12):
            count = count_map.get(code, 0)
            print(
                f"Class {code:>2} - {SCL_NAMES[code]:<25} "
                f"{count:>8} pixels  ({count / total * 100:6.3f}%)"
            )
        unknown = sorted(set(count_map) - set(SCL_NAMES))
        if unknown:
            print("Unexpected SCL values:", unknown)


def main():
    token = get_access_token()
    bbox = make_bbox()
    print("Diagnostic output:", OUTPUT_DIR)
    print("CRS:", TARGET_CRS, "| BBox:", bbox)
    for label, date in DATES.items():
        path = retrieve_one(token, label, date, bbox)
        report_scl(path, label)
    print("\nSCL diagnostic completed. Existing production/test rasters were not changed.")


if __name__ == "__main__":
    main()
