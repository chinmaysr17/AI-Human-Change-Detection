
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

OUTPUT = ROOT / "dataset/Mining/component_chips"
OUTPUT.mkdir(parents=True, exist_ok=True)

STAC_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1/"
    "collections/sentinel-2-l2a/items/"
)

BANDS = ["B02", "B03", "B04", "B08"]

CHIP_SIZE = 256
MIN_COMPONENT_AREA = 0.000025


def decode_geometry(blob):
    flags = blob[3]
    code = (flags >> 1) & 7
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


def extract_tile(tile_id, product_id, mining_type, activity):
    print(f"\nProcessing tile {tile_id}")

    polygon = get_polygon(tile_id)

    response = requests.get(STAC_URL + product_id, timeout=60)
    response.raise_for_status()

    item = planetary_computer.sign(response.json())
    assets = item["assets"]

    for band in BANDS:
        if band not in assets:
            raise RuntimeError(f"Missing asset: {band}")

    with rasterio.open(assets["B04"]["href"]) as reference:
        projected = shape(
            transform_geom(
                "EPSG:4326",
                reference.crs,
                mapping(polygon)
            )
        )

        if projected.geom_type == "Polygon":
            components = [projected]
        else:
            components = list(projected.geoms)

        components = [
            component for component in components
            if component.area >= MIN_COMPONENT_AREA
        ]

        print("Eligible components:", len(components))

        if not components:
            raise RuntimeError("No components passed area filter.")

        # Read the product's raster window separately for each component.
        crs = reference.crs

        component_windows = []

        for index, component in enumerate(components, start=1):
            center_x, center_y = (
                component.representative_point().coords[0]
            )

            row, col = reference.index(center_x, center_y)

            window = Window(
                col - CHIP_SIZE // 2,
                row - CHIP_SIZE // 2,
                CHIP_SIZE,
                CHIP_SIZE
            )

            transform = reference.window_transform(window)

            component_windows.append(
                (index, component, window, transform)
            )

    saved = 0

    for index, component, window, transform in component_windows:
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

        mask = rasterize(
            [(mapping(component), 1)],
            out_shape=(CHIP_SIZE, CHIP_SIZE),
            transform=transform,
            fill=0,
            dtype="uint8"
        )

        coverage = float(mask.mean() * 100)

        if mask.sum() == 0:
            continue

        # Keep the output traceable to the source tile and component.
        prefix = f"tile_{tile_id}_component_{index}"

        np.save(OUTPUT / f"{prefix}_image.npy", image)
        np.save(OUTPUT / f"{prefix}_mask.npy", mask)

        saved += 1

        print(
            f"  Component {index}: "
            f"coverage={coverage:.2f}%"
        )

    return saved


def main():
    with open(CSV_FILE, newline="", encoding="utf-8") as file:
        samples = list(csv.DictReader(file))

    total_saved = 0
    failed = []

    for sample in samples:
        tile_id = int(sample["tile_id"])

        try:
            count = extract_tile(
                tile_id,
                sample["s2_tile_id"],
                sample["mining_type"],
                sample["activity_type"]
            )

            total_saved += count

        except Exception as error:
            print(f"FAILED tile {tile_id}: {error}")
            failed.append(tile_id)

    print("\nCOMPONENT EXTRACTION SUMMARY")
    print("Source tiles:", len(samples))
    print("Component chips saved:", total_saved)
    print("Failed tiles:", failed)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
