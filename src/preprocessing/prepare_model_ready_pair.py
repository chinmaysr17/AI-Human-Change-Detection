
import os
import numpy as np
import rasterio

INPUTS = {
    "before": (
        "data/sentinel2/processed/calibrated/"
        "sentinel2_20260927_shivamogga_10km_reflectance.tif"
    ),
    "after": (
        "data/sentinel2/processed/calibrated/"
        "sentinel2_20261027_shivamogga_10km_reflectance.tif"
    ),
}

OUTPUT_DIR = "data/sentinel2/processed/model_ready"

BAND_NAMES = [
    "Before_B02_Blue",
    "Before_B03_Green",
    "Before_B04_Red",
    "Before_B08_NIR",
    "After_B02_Blue",
    "After_B03_Green",
    "After_B04_Red",
    "After_B08_NIR",
]

NODATA = -9999.0


def main():
    arrays = {}
    profiles = {}

    for date_name, path in INPUTS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with rasterio.open(path) as src:
            arrays[date_name] = src.read().astype(np.float32)
            profiles[date_name] = src.profile.copy()

    before = arrays["before"]
    after = arrays["after"]

    before_profile = profiles["before"]
    after_profile = profiles["after"]

    if before.shape != after.shape:
        raise ValueError("Before and after image dimensions differ.")

    if before_profile["transform"] != after_profile["transform"]:
        raise ValueError("Image transforms do not match.")

    if before_profile["crs"] != after_profile["crs"]:
        raise ValueError("Image coordinate systems do not match.")

    # A pixel is valid only if all bands on both dates are valid.
    valid_before = np.all(before != NODATA, axis=0)
    valid_after = np.all(after != NODATA, axis=0)
    common_valid = valid_before & valid_after

    # Stack the two dates: 4 before bands + 4 after bands.
    stacked = np.concatenate([before, after], axis=0)

    # Use the same NoData mask for all eight bands.
    stacked[:, ~common_valid] = NODATA

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        "sentinel2_shivamogga_20260927_20261027_8band.tif"
    )

    profile = before_profile.copy()
    profile.update(
        driver="GTiff",
        count=8,
        dtype="float32",
        nodata=NODATA,
        compress="deflate"
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stacked)

        for index, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(index, name)

    print("=" * 60)
    print("MODEL-READY IMAGE PAIR")
    print("=" * 60)
    print("Output:", output_path)
    print("Dimensions:", before.shape[2], "x", before.shape[1])
    print("Input channels:", stacked.shape[0])
    print("Common valid pixels:", int(common_valid.sum()))
    print("Invalid pixels:", int((~common_valid).sum()))
    print("Common valid percentage:", f"{common_valid.mean() * 100:.2f}%")
    print("Band order:")

    for index, name in enumerate(BAND_NAMES, start=1):
        print(f"{index}. {name}")

    print("\nImage alignment: PASS")
    print("Eight-band stacking: PASS")
    print("Model-ready pair preparation: PASS")


if __name__ == "__main__":
    main()
