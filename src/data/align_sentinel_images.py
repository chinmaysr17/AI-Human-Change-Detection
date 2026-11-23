"""
Align a BEFORE/AFTER Sentinel-2 GeoTIFF pair to one common pixel grid.

Usage:
    python src/data/align_sentinel_images.py ^
        data/sentinel2/geotiff_test/before.tif ^
        data/sentinel2/geotiff_test/after.tif ^
        data/sentinel2/aligned

The BEFORE image defines the target grid. AFTER is reprojected only when
its CRS, transform, dimensions, or bounds differ.
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def same_grid(a, b, tolerance=1e-8):
    return (
        a.crs == b.crs
        and a.width == b.width
        and a.height == b.height
        and np.allclose(
            tuple(a.transform),
            tuple(b.transform),
            rtol=0,
            atol=tolerance,
        )
    )


def align_pair(before_path, after_path, output_dir):
    before_path = Path(before_path)
    after_path = Path(after_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before_out = output_dir / "before_aligned.tif"
    after_out = output_dir / "after_aligned.tif"

    with rasterio.open(before_path) as before, rasterio.open(after_path) as after:
        if before.crs is None or after.crs is None:
            raise ValueError("Both inputs must have a defined CRS.")

        if before.count != after.count:
            raise ValueError(
                f"Band-count mismatch: BEFORE={before.count}, AFTER={after.count}"
            )

        if before.count != 3:
            raise ValueError("Expected three bands ordered as B04, B03, B02.")

        if before.crs.to_string() != "EPSG:32643":
            raise ValueError(
                f"Unexpected BEFORE CRS {before.crs}; expected EPSG:32643."
            )

        target_profile = before.profile.copy()
        target_profile.update(
            driver="GTiff",
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        before_data = before.read()
        with rasterio.open(before_out, "w", **target_profile) as dst:
            dst.write(before_data)
            dst.set_band_description(1, "B04_Red")
            dst.set_band_description(2, "B03_Green")
            dst.set_band_description(3, "B02_Blue")

        if same_grid(before, after):
            after_data = after.read()
        else:
            after_data = np.full(
                (after.count, before.height, before.width),
                before.nodata if before.nodata is not None else -9999.0,
                dtype=np.dtype(before.dtypes[0]),
            )

            for band_index in range(1, before.count + 1):
                reproject(
                    source=rasterio.band(after, band_index),
                    destination=after_data[band_index - 1],
                    src_transform=after.transform,
                    src_crs=after.crs,
                    src_nodata=after.nodata,
                    dst_transform=before.transform,
                    dst_crs=before.crs,
                    dst_nodata=(
                        before.nodata
                        if before.nodata is not None
                        else -9999.0
                    ),
                    resampling=Resampling.bilinear,
                )

        with rasterio.open(after_out, "w", **target_profile) as dst:
            dst.write(after_data)
            dst.set_band_description(1, "B04_Red")
            dst.set_band_description(2, "B03_Green")
            dst.set_band_description(3, "B02_Blue")

    with rasterio.open(before_out) as a, rasterio.open(after_out) as b:
        if not same_grid(a, b):
            raise RuntimeError("Alignment validation failed: output grids differ.")

        print("Alignment validation passed.")
        print("CRS:", a.crs)
        print("Dimensions:", a.width, "x", a.height)
        print("Bounds:", a.bounds)
        print("Transform:", a.transform)
        print("BEFORE:", before_out.resolve())
        print("AFTER:", after_out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    align_pair(args.before, args.after, args.output_dir)
