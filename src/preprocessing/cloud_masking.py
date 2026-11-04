
import os
import glob
import rasterio
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_DIR = "data/sentinel2/processed"
OUTPUT_DIR = "data/sentinel2/processed/masked"

# SCL classes to exclude
INVALID_CLASSES = [0, 1, 3, 7, 8, 9, 10, 11]

BAND_NAMES = [
    "B02_Blue",
    "B03_Green",
    "B04_Red",
    "B08_NIR"
]


# --------------------------------------------------
# APPLY CLOUD AND SHADOW MASK
# --------------------------------------------------

def apply_mask(filepath):

    filename = os.path.basename(filepath)

    print("\nProcessing:", filename)

    with rasterio.open(filepath) as src:

        data = src.read()

        # First four bands are spectral bands
        spectral = data[:4]

        # Fifth band is SCL
        scl = data[4]

        # Identify invalid pixels
        invalid_mask = np.isin(
            scl,
            INVALID_CLASSES
        )

        valid_mask = ~invalid_mask

        total_pixels = scl.size
        invalid_pixels = np.count_nonzero(invalid_mask)
        valid_pixels = np.count_nonzero(valid_mask)

        invalid_percentage = (
            invalid_pixels / total_pixels
        ) * 100

        valid_percentage = (
            valid_pixels / total_pixels
        ) * 100

        print("Total pixels:", total_pixels)
        print("Valid pixels:", valid_pixels)
        print("Masked pixels:", invalid_pixels)
        print(
            "Valid percentage:",
            round(valid_percentage, 2), "%"
        )
        print(
            "Masked percentage:",
            round(invalid_percentage, 2), "%"
        )

        # Preserve spectral values and store masked pixels
        # as zero in the output.
        masked_data = spectral.copy()

        masked_data[:, invalid_mask] = 0

        profile = src.profile.copy()

        profile.update(
            count=4,
            dtype="uint16",
            nodata=0,
            compress="deflate"
        )

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename.replace(".tif", "_masked.tif")
        )

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:

            dst.write(masked_data)

            for index, name in enumerate(
                BAND_NAMES,
                start=1
            ):
                dst.set_band_description(
                    index,
                    name
                )

        print("Saved:", output_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    files = sorted(
        glob.glob(
            os.path.join(INPUT_DIR, "*.tif")
        )
    )

    if not files:
        raise FileNotFoundError(
            "No processed GeoTIFF files found."
        )

    for filepath in files:
        apply_mask(filepath)

    print("\nCloud and shadow masking completed.")
