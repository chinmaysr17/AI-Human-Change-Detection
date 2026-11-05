
import json
from pathlib import Path

import planetary_computer
import rasterio
import requests
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]

PRODUCT_ID = (
    "S2A_MSIL2A_20191017T004711_R102_T54LWM_20201004T031000"
)

OUTPUT = ROOT / "data" / "sentinel2" / "mining"
OUTPUT.mkdir(parents=True, exist_ok=True)

STAC_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1/"
    f"collections/sentinel-2-l2a/items/{PRODUCT_ID}"
)

BANDS = ["B02", "B03", "B04", "B08"]


def main():
    print("Fetching Sentinel-2 STAC metadata...")

    response = requests.get(STAC_URL, timeout=60)
    response.raise_for_status()
    item = response.json()

    signed_item = planetary_computer.sign(item)

    print("Product:", item["id"])

    # Read the mining polygon from the GeoPackage.
    import sqlite3
    from shapely import wkb

    gpkg = ROOT / "dataset/Mining/extracted/mining_area_data.gpkg"

    connection = sqlite3.connect(gpkg)
    row = connection.execute(
        "SELECT geom FROM preferred_polygons WHERE tile_id = 2805 LIMIT 1"
    ).fetchone()
    connection.close()

    if row is None:
        raise RuntimeError("Mining polygon not found.")

    blob = row[0]
    envelope_code = (blob[3] >> 1) & 0b111
    envelope_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    polygon = wkb.loads(blob[8 + envelope_sizes[envelope_code]:])

    # Locate the raster asset links.
    assets = signed_item["assets"]

    for band in BANDS:
        if band not in assets:
            raise RuntimeError(f"Missing STAC asset: {band}")

    # Download a small window around the polygon.
    # Use B04 to calculate the matching raster window.
    with rasterio.open(assets["B04"]["href"]) as src:
        from rasterio.warp import transform_geom
        from shapely.geometry import mapping, shape

        projected = shape(
            transform_geom(
                "EPSG:4326",
                src.crs,
                mapping(polygon)
            )
        )

        if not projected.intersects(shape({
            "type": "Polygon",
            "coordinates": [[
                (src.bounds.left, src.bounds.bottom),
                (src.bounds.right, src.bounds.bottom),
                (src.bounds.right, src.bounds.top),
                (src.bounds.left, src.bounds.top),
                (src.bounds.left, src.bounds.bottom)
            ]]
        })):
            raise RuntimeError(
                "Polygon does not overlap this Sentinel-2 product."
            )

        cx, cy = projected.centroid.coords[0]
        center_row, center_col = src.index(cx, cy)

        size = 256
        window = Window(
            center_col - size // 2,
            center_row - size // 2,
            size,
            size
        )

        transform = src.window_transform(window)

    print("Extracting 256 x 256 chips...")

    for band in BANDS:
        output_file = OUTPUT / f"mining_2805_{band}.tif"

        with rasterio.open(assets[band]["href"]) as src:
            data = src.read(
                1,
                window=window,
                boundless=True,
                fill_value=0
            )

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=size,
                width=size,
                transform=transform,
                count=1,
                compress="lzw"
            )

            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(data, 1)

        print("Saved:", output_file.name)

    print("\nDownload and extraction completed.")


if __name__ == "__main__":
    main()
