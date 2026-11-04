
import os
import numpy as np
import rasterio

INPUTS = {
    "20260927": (
        "data/sentinel2/processed/masked/"
        "sentinel2_20260927_shivamogga_10km_masked.tif"
    ),
    "20261027": (
        "data/sentinel2/processed/masked/"
        "sentinel2_20261027_shivamogga_10km_masked.tif"
    ),
}

OUTPUT_DIR = "data/sentinel2/processed/calibrated"

QUANTIFICATION_VALUE = 10000.0
BOA_OFFSET = -1000.0
NODATA_VALUE = -9999.0

BAND_NAMES = ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR"]


def calibrate_image(date, input_path):
    print("\n" + "=" * 55)
    print(f"CALIBRATING IMAGE: {date}")
    print("=" * 55)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    with rasterio.open(input_path) as src:
        raw = src.read().astype(np.float32)
        profile = src.profile.copy()

        # Cloud-masked pixels were stored as zero.
        valid_mask = np.all(raw > 0, axis=0)

        # BOA reflectance = (DN + additive offset) / quantification value
        reflectance = (
            raw + BOA_OFFSET
        ) / QUANTIFICATION_VALUE

        # Preserve invalid pixels as NoData.
        reflectance[:, ~valid_mask] = NODATA_VALUE

        profile.update(
            driver="GTiff",
            dtype="float32",
            count=4,
            nodata=NODATA_VALUE,
            compress="deflate"
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output_path = os.path.join(
            OUTPUT_DIR,
            f"sentinel2_{date}_shivamogga_10km_reflectance.tif"
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(reflectance)

            for index, name in enumerate(BAND_NAMES, start=1):
                dst.set_band_description(index, name)

        print("Output:", output_path)
        print("Dimensions:", src.width, "x", src.height)
        print("Valid pixels:", int(valid_mask.sum()))
        print("Invalid pixels:", int((~valid_mask).sum()))

        for index, name in enumerate(BAND_NAMES):
            values = reflectance[index][valid_mask]

            print(
                f"{name}: "
                f"Min={values.min():.4f}, "
                f"Max={values.max():.4f}, "
                f"Mean={values.mean():.4f}, "
                f"Median={np.median(values):.4f}"
            )


if __name__ == "__main__":
    for date, input_path in INPUTS.items():
        calibrate_image(date, input_path)

    print("\nSurface reflectance calibration completed.")
