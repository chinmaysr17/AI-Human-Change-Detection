
import os
import glob
import rasterio
import numpy as np

from rasterio.windows import Window
from rasterio.warp import reproject, Resampling


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = "data/sentinel2/raw"
OUTPUT_DIR = "data/sentinel2/processed"

CENTER_LAT = 13.9299
CENTER_LON = 75.5681

# Study area: 10 km x 10 km
LEFT = 556369.01
RIGHT = 566369.01
BOTTOM = 1535046.86
TOP = 1545046.86

PIXEL_SIZE = 10
WIDTH = 1000
HEIGHT = 1000

SPECTRAL_BANDS = ["B02", "B03", "B04", "B08"]

DATES = ["20260927", "20261027"]


# --------------------------------------------------
# FIND SATELLITE BAND
# --------------------------------------------------

def find_band(product_path, band, resolution):

    pattern = os.path.join(
        product_path,
        "**",
        "IMG_DATA",
        resolution,
        f"*_{band}_{resolution[1:]}.jp2"
    )

    matches = glob.glob(pattern, recursive=True)

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {band} file in {product_path}; "
            f"found {len(matches)}"
        )

    return matches[0]


# --------------------------------------------------
# PROCESS ONE SATELLITE PRODUCT
# --------------------------------------------------

def process_product(date):

    print(f"\nProcessing Sentinel-2: {date}")

    products = [
        name for name in os.listdir(BASE_DIR)
        if name.endswith(".SAFE") and date in name
    ]

    if len(products) != 1:
        raise FileNotFoundError(
            f"Expected one product for {date}; "
            f"found {len(products)}"
        )

    product_path = os.path.join(BASE_DIR, products[0])

    # ----------------------------------------------
    # REFERENCE GRID AND EXACT PIXEL ALIGNMENT
    # ----------------------------------------------

    reference_path = find_band(
        product_path, "B04", "R10m"
    )

    with rasterio.open(reference_path) as reference:

        col = round(
            (LEFT - reference.bounds.left) / PIXEL_SIZE
        )

        row = round(
            (reference.bounds.top - TOP) / PIXEL_SIZE
        )

        window = Window(
            col,
            row,
            WIDTH,
            HEIGHT
        )

        target_transform = reference.window_transform(window)

        # Ensure the requested crop is inside the image
        if (
            col < 0 or row < 0
            or col + WIDTH > reference.width
            or row + HEIGHT > reference.height
        ):
            raise ValueError("Study area is outside the satellite image")

    print("Pixel-grid alignment verified")
    print("Crop window:", col, row, WIDTH, HEIGHT)

    # ----------------------------------------------
    # READ FOUR 10M SPECTRAL BANDS
    # ----------------------------------------------

    arrays = []

    for band in SPECTRAL_BANDS:

        path = find_band(
            product_path, band, "R10m"
        )

        with rasterio.open(path) as src:

            data = src.read(
                1,
                window=window
            )

            if data.shape != (HEIGHT, WIDTH):
                raise ValueError(
                    f"Unexpected crop size for {band}: {data.shape}"
                )

            arrays.append(data)

        print(f"{band}: crop successful")

    # ----------------------------------------------
    # READ AND ALIGN 20M SCL TO 10M GRID
    # ----------------------------------------------

    scl_path = find_band(
        product_path, "SCL", "R20m"
    )

    scl = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.uint16
    )

    with rasterio.open(scl_path) as src:

        reproject(
            source=rasterio.band(src, 1),
            destination=scl,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs="EPSG:32643",
            resampling=Resampling.nearest
        )

    print("SCL: aligned to 10m grid")

    arrays.append(scl)

    # ----------------------------------------------
    # SAVE FIVE-BAND GEOTIFF
    # ----------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sentinel2_{date}_shivamogga_10km.tif"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with rasterio.open(reference_path) as reference:

        profile = reference.profile.copy()

    profile.update(
        driver="GTiff",
        width=WIDTH,
        height=HEIGHT,
        count=5,
        dtype="uint16",
        crs="EPSG:32643",
        transform=target_transform,
        compress="deflate"
    )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        for index, data in enumerate(arrays, start=1):
            dst.write(
                data.astype("uint16"),
                index
            )

        dst.set_band_description(1, "B02_Blue")
        dst.set_band_description(2, "B03_Green")
        dst.set_band_description(3, "B04_Red")
        dst.set_band_description(4, "B08_NIR")
        dst.set_band_description(5, "SCL_Classification")

    print("Saved:", output_path)
    print("Dimensions:", WIDTH, "x", HEIGHT)
    print("Bands:", len(arrays))


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    for date in DATES:
        process_product(date)

    print("\nBoth Sentinel-2 products processed successfully.")
