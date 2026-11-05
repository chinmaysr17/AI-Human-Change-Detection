
import csv
import sqlite3
from pathlib import Path

import numpy as np
import planetary_computer
import rasterio
import requests

from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.warp import transform_geom
from shapely import wkb
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[2]

GPKG = ROOT / "dataset/Mining/extracted/mining_area_data.gpkg"
CSV_FILE = ROOT / "dataset/Mining/mining_training_30.csv"

OUTPUT = ROOT / "dataset/Mining/training_chips_30"
OUTPUT.mkdir(parents=True, exist_ok=True)

STAC_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1/"
    "collections/sentinel-2-l2a/items/"
)

BANDS = ["B02", "B03", "B04", "B08"]
SIZE = 256


def decode_geometry(blob):
    flags = blob[3]
    code = (flags >> 1) & 0b111
    sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    return wkb.loads(blob[8 + sizes[code]:])


def get_polygon(tile_id):
    connection = sqlite3.connect(GPKG)

    row = connection.execute(
        "SELECT geom FROM preferred_polygons "
        "WHERE tile_id = ? LIMIT 1",
        (tile_id,)
    ).fetchone()

    connection.close()

    if row is None:
        raise RuntimeError("Mining polygon not found.")

    return decode_geometry(row[0])


def extract_sample(tile_id, product_id):
    print(f"\nProcessing tile {tile_id}")

    polygon = get_polygon(tile_id)

    response = requests.get(
        STAC_URL + product_id,
        timeout=60
    )
    response.raise_for_status()

    item = planetary_computer.sign(response.json())
    assets = item["assets"]

    for band in BANDS:
        if band not in assets:
            raise RuntimeError(f"Missing asset: {band}")

    # Locate the mining polygon in the B04 raster.
    with rasterio.open(assets["B04"]["href"]) as src:
        projected = shape(
            transform_geom(
                "EPSG:4326",
                src.crs,
                mapping(polygon)
            )
        )

        raster_bounds = shape({
            "type": "Polygon",
            "coordinates": [[
                (src.bounds.left, src.bounds.bottom),
                (src.bounds.right, src.bounds.bottom),
                (src.bounds.right, src.bounds.top),
                (src.bounds.left, src.bounds.top),
                (src.bounds.left, src.bounds.bottom)
            ]]
        })

        if not projected.intersects(raster_bounds):
            raise RuntimeError("Polygon does not overlap raster.")

        x, y = projected.centroid.coords[0]
        row, col = src.index(x, y)

        window = Window(
            col - SIZE // 2,
            row - SIZE // 2,
            SIZE,
            SIZE
        )

        transform = src.window_transform(window)
        crs = src.crs

    # Extract four spectral bands.
    image_bands = []

    for band in BANDS:
        with rasterio.open(assets[band]["href"]) as src:
            if src.crs != crs:
                raise RuntimeError(f"CRS mismatch: {band}")

            data = src.read(
                1,
                window=window,
                boundless=True,
                fill_value=0
            )

            image_bands.append(data)

    image = np.stack(image_bands, axis=-1)

    # Generate binary mining mask.
    mask = rasterize(
        [(mapping(projected), 1)],
        out_shape=(SIZE, SIZE),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    if mask.sum() == 0:
        raise RuntimeError("Empty mining mask.")

    np.save(OUTPUT / f"mining_{tile_id}.npy", image)
    np.save(OUTPUT / f"mask_{tile_id}.npy", mask)

    coverage = float(mask.mean() * 100)

    print("Image:", image.shape)
    print("Mask:", mask.shape)
    print(f"Mining coverage: {coverage:.2f}%")

    return coverage


def main():
    with open(CSV_FILE, newline="", encoding="utf-8") as file:
        samples = list(csv.DictReader(file))

    print("Total selected samples:", len(samples))

    results = []
    failed = []

    for sample in samples:
        tile_id = int(sample["tile_id"])
        product_id = sample["s2_tile_id"]

        try:
            coverage = extract_sample(tile_id, product_id)

            results.append({
                "tile_id": tile_id,
                "mining_type": sample["mining_type"],
                "activity_type": sample["activity_type"],
                "coverage_percent": round(coverage, 2),
                "status": "success"
            })

        except Exception as error:
            print(f"FAILED tile {tile_id}: {error}")
            failed.append(tile_id)

    report = OUTPUT / "extraction_report.csv"

    with open(report, "w", newline="", encoding="utf-8") as file:
        fields = [
            "tile_id",
            "mining_type",
            "activity_type",
            "coverage_percent",
            "status"
        ]

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

        for tile_id in failed:
            writer.writerow({
                "tile_id": tile_id,
                "status": "failed"
            })

    print("\nEXTRACTION SUMMARY")
    print("Successful:", len(results))
    print("Failed:", len(failed))
    print("Report:", report)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
