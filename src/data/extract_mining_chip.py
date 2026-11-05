
import sqlite3
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.warp import transform_geom
from shapely import wkb
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[2]

GPKG = ROOT / "dataset/Mining/extracted/mining_area_data.gpkg"
OUTPUT = ROOT / "dataset/Mining/chips"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Use the Sentinel-2 product matched to tile ID 2805.
PRODUCT_ID = "S2A_MSIL2A_20191017T004711_R102_T54LWM_20201004T031000"

RASTER_DIR = ROOT / "data/sentinel2/mining"
BANDS = ["B02", "B03", "B04", "B08"]


def read_gpkg_geometry(blob):
    """Decode a GeoPackage geometry blob into a Shapely geometry."""
    flags = blob[3]
    envelope_code = (flags >> 1) & 0b111
    envelope_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}

    offset = 8 + envelope_sizes[envelope_code]
    return wkb.loads(blob[offset:])


def main():
    # Find the matching tile and its preferred mining polygon.
    connection = sqlite3.connect(GPKG)
    connection.row_factory = sqlite3.Row

    tile = connection.execute(
        "SELECT tile_id FROM tiles WHERE s2_tile_id = ?",
        (PRODUCT_ID,)
    ).fetchone()

    if tile is None:
        raise RuntimeError("Product ID not found in the GeoPackage.")

    tile_id = tile["tile_id"]

    polygon_row = connection.execute(
        "SELECT geom FROM preferred_polygons WHERE tile_id = ? LIMIT 1",
        (tile_id,)
    ).fetchone()

    connection.close()

    if polygon_row is None:
        raise RuntimeError("No preferred mining polygon found.")

    polygon = read_gpkg_geometry(polygon_row["geom"])

    print("Tile ID:", tile_id)
    print("Polygon type:", polygon.geom_type)
    print("Polygon bounds:", polygon.bounds)

    # Locate the four downloaded Sentinel-2 band files.
    band_paths = {}

    for band in BANDS:
        matches = list(RASTER_DIR.rglob(f"*{band}*.tif"))
        if not matches:
            raise FileNotFoundError(
                f"Missing {band} raster under {RASTER_DIR}"
            )
        band_paths[band] = matches[0]

    # Transform the polygon to the raster CRS.
    with rasterio.open(band_paths["B04"]) as reference:
        polygon_projected = shape(
            transform_geom(
                "EPSG:4326",
                reference.crs,
                mapping(polygon)
            )
        )

        if not polygon_projected.intersects(shape({
            "type": "Polygon",
            "coordinates": [[
                (reference.bounds.left, reference.bounds.bottom),
                (reference.bounds.right, reference.bounds.bottom),
                (reference.bounds.right, reference.bounds.top),
                (reference.bounds.left, reference.bounds.top),
                (reference.bounds.left, reference.bounds.bottom)
            ]]
        })):
            raise RuntimeError(
                "Mining polygon does not overlap the selected raster. "
                "Check that the raster belongs to the matched product."
            )

        center_x, center_y = polygon_projected.centroid.coords[0]
        center_row, center_col = reference.index(center_x, center_y)

        chip_size = 256
        row_start = center_row - chip_size // 2
        col_start = center_col - chip_size // 2

        window = Window(
            col_start, row_start, chip_size, chip_size
        )

        window_transform = reference.window_transform(window)

    # Read all bands using the same pixel window.
    image = []

    for band in BANDS:
        with rasterio.open(band_paths[band]) as src:
            if src.crs != reference.crs:
                raise RuntimeError("Band CRS mismatch.")

            data = src.read(
                1,
                window=window,
                boundless=True,
                fill_value=0
            )
            image.append(data)

    image = np.stack(image, axis=-1)

    # Create a binary mask: 1 = mining polygon, 0 = background.
    mask = rasterize(
        [(mapping(polygon_projected), 1)],
        out_shape=(chip_size, chip_size),
        transform=window_transform,
        fill=0,
        dtype="uint8"
    )

    if mask.sum() == 0:
        raise RuntimeError(
            "The extracted chip contains no mining polygon pixels. "
            "Check product and polygon alignment."
        )

    np.save(OUTPUT / "mining_chip_2805.npy", image)
    np.save(OUTPUT / "mining_mask_2805.npy", mask)

    print("\nEXTRACTION SUCCESSFUL")
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Mining pixels:", int(mask.sum()))
    print("Image saved:", OUTPUT / "mining_chip_2805.npy")
    print("Mask saved:", OUTPUT / "mining_mask_2805.npy")


if __name__ == "__main__":
    main()
