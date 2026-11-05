
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
OUTPUT = ROOT / "dataset/Mining/training_chips"

OUTPUT.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/"

BANDS = ["B02", "B03", "B04", "B08"]
CHIP_SIZE = 256


def decode_geometry(blob):
    flags = blob[3]
    envelope_code = (flags >> 1) & 0b111
    envelope_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    return wkb.loads(blob[8 + envelope_sizes[envelope_code]:])


def get_samples():
    connection = sqlite3.connect(GPKG)

    rows = connection.execute("""
        SELECT tile_id, s2_tile_id
        FROM tiles
        WHERE split = 'train'
        AND source_dataset != 'rejected'
        LIMIT 5
    """).fetchall()

    connection.close()
    return rows


def extract_sample(tile_id, product_id):
    print(f"\nProcessing tile {tile_id}: {product_id}")

    # Read the corresponding mining polygon.
    connection = sqlite3.connect(GPKG)

    row = connection.execute(
        "SELECT geom FROM preferred_polygons WHERE tile_id = ? LIMIT 1",
        (tile_id,)
    ).fetchone()

    connection.close()

    if row is None:
        raise RuntimeError(f"No polygon for tile {tile_id}")

    polygon = decode_geometry(row[0])

    # Retrieve and sign Sentinel-2 STAC metadata.
    response = requests.get(STAC_URL + product_id, timeout=60)
    response.raise_for_status()

    item = planetary_computer.sign(response.json())
    assets = item["assets"]

    for band in BANDS:
        if band not in assets:
            raise RuntimeError(f"{band} missing for tile {tile_id}")

    # Find polygon location in the B04 raster.
    with rasterio.open(assets["B04"]["href"]) as reference:
        projected = shape(
            transform_geom(
                "EPSG:4326",
                reference.crs,
                mapping(polygon)
            )
        )

        if not projected.intersects(
            shape({
                "type": "Polygon",
                "coordinates": [[
                    (reference.bounds.left, reference.bounds.bottom),
                    (reference.bounds.right, reference.bounds.bottom),
                    (reference.bounds.right, reference.bounds.top),
                    (reference.bounds.left, reference.bounds.top),
                    (reference.bounds.left, reference.bounds.bottom)
                ]]
            })
        ):
            raise RuntimeError(
                f"Polygon does not overlap raster for tile {tile_id}"
            )

        center_x, center_y = projected.centroid.coords[0]
        center_row, center_col = reference.index(center_x, center_y)

        window = Window(
            center_col - CHIP_SIZE // 2,
            center_row - CHIP_SIZE // 2,
            CHIP_SIZE,
            CHIP_SIZE
        )

        transform = reference.window_transform(window)
        crs = reference.crs

    # Extract the four bands.
    image_bands = []

    for band in BANDS:
        with rasterio.open(assets[band]["href"]) as src:
            data = src.read(
                1,
                window=window,
                boundless=True,
                fill_value=0
            )
            image_bands.append(data)

    image = np.stack(image_bands, axis=-1)

    # Rasterize the mining polygon.
    mask = rasterize(
        [(mapping(projected), 1)],
        out_shape=(CHIP_SIZE, CHIP_SIZE),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    if mask.sum() == 0:
        raise RuntimeError(f"Empty mining mask for tile {tile_id}")

    # Save image and ground-truth mask.
    np.save(OUTPUT / f"mining_{tile_id}.npy", image)
    np.save(OUTPUT / f"mask_{tile_id}.npy", mask)

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Mining coverage:", round(mask.mean() * 100, 2), "%")
    print("Saved successfully.")


def main():
    samples = get_samples()

    print("Training samples found:", len(samples))

    for tile_id, product_id in samples:
        try:
            extract_sample(tile_id, product_id)
        except Exception as error:
            print(f"FAILED tile {tile_id}: {error}")

    print("\nMining training extraction finished.")
    print("Output folder:", OUTPUT)


if __name__ == "__main__":
    main()
