
import os
import csv
import numpy as np
import rasterio

FILES = {
    "20260927": (
        "data/sentinel2/processed/calibrated/"
        "sentinel2_20260927_shivamogga_10km_reflectance.tif"
    ),
    "20261027": (
        "data/sentinel2/processed/calibrated/"
        "sentinel2_20261027_shivamogga_10km_reflectance.tif"
    ),
}

BANDS = ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR"]


def load_image(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)

    return data, nodata, transform, crs, shape


def main():
    print("=" * 65)
    print("CALIBRATED REFLECTANCE DISTRIBUTION COMPARISON")
    print("=" * 65)

    images = {}

    for date, path in FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        images[date] = load_image(path)

    before, nodata_before, transform_before, crs_before, shape_before = (
        images["20260927"]
    )
    after, nodata_after, transform_after, crs_after, shape_after = (
        images["20261027"]
    )

    if shape_before != shape_after:
        raise ValueError("Image dimensions do not match.")

    if transform_before != transform_after or crs_before != crs_after:
        raise ValueError("Image grids are not aligned.")

    # A pixel is valid only when all four bands are valid on both dates.
    valid_before = np.all(before != nodata_before, axis=0)
    valid_after = np.all(after != nodata_after, axis=0)
    common_valid = valid_before & valid_after

    print("\nIMAGE ALIGNMENT: PASS")
    print("Image dimensions:", shape_before)
    print("Common valid pixels:", int(common_valid.sum()))
    print(
        "Common valid percentage:",
        f"{common_valid.mean() * 100:.2f}%"
    )

    results = []

    for band_index, band_name in enumerate(BANDS):
        old_values = before[band_index][common_valid]
        new_values = after[band_index][common_valid]

        print("\n" + "-" * 55)
        print(band_name)
        print("-" * 55)

        for date, values in [
            ("2026-09-27", old_values),
            ("2026-10-27", new_values),
        ]:
            p = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])

            negative = int(np.sum(values < 0))
            above_one = int(np.sum(values > 1))

            print(f"\n{date}")
            print(f"Minimum: {values.min():.4f}")
            print(f"Maximum: {values.max():.4f}")
            print(f"Mean: {values.mean():.4f}")
            print(f"Standard deviation: {values.std():.4f}")
            print("Percentiles 1, 5, 25, 50, 75, 95, 99:")
            print(np.round(p, 4))
            print(f"Negative values: {negative}")
            print(f"Values above 1.0: {above_one}")

            results.append({
                "date": date,
                "band": band_name,
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "negative_pixels": negative,
                "above_one_pixels": above_one,
            })

        print("\nDifference between date means:")
        print(f"{new_values.mean() - old_values.mean():+.5f}")

        print("Difference between date medians:")
        print(
            f"{np.median(new_values) - np.median(old_values):+.5f}"
        )

    output_path = (
        "data/sentinel2/processed/"
        "reflectance_distribution_comparison.csv"
    )

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("COMPARISON COMPLETED")
    print("CSV report:", output_path)
    print("=" * 65)


if __name__ == "__main__":
    main()
