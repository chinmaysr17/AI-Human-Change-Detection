
import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt


INPUT_DIR = "data/sentinel2/processed/masked"
OUTPUT_DIR = "data/sentinel2/processed/previews"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def stretch_band(band, valid_mask):

    values = band[valid_mask]

    low, high = np.percentile(values, [2, 98])

    stretched = np.clip(
        (band.astype(np.float32) - low) / (high - low),
        0,
        1
    )

    return stretched


def create_previews(filepath):

    filename = os.path.basename(filepath)

    print("\nCreating previews:", filename)

    with rasterio.open(filepath) as src:

        data = src.read()

        # Identify pixels where all four bands are valid
        valid_mask = np.all(data > 0, axis=0)

        if not np.any(valid_mask):
            raise ValueError("No valid pixels found.")

        blue = stretch_band(data[0], valid_mask)
        green = stretch_band(data[1], valid_mask)
        red = stretch_band(data[2], valid_mask)
        nir = stretch_band(data[3], valid_mask)

        # Natural colour: Red, Green, Blue
        natural = np.dstack((red, green, blue))

        # False colour: NIR, Red, Green
        false_color = np.dstack((nir, red, green))

        natural[~valid_mask] = 0
        false_color[~valid_mask] = 0

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 7)
        )

        axes[0].imshow(natural)
        axes[0].set_title("Natural Colour (RGB)")
        axes[0].axis("off")

        axes[1].imshow(false_color)
        axes[1].set_title("False Colour (NIR-R-G)")
        axes[1].axis("off")

        date = filename.split("_")[1]

        fig.suptitle(
            f"Shivamogga Sentinel-2 - {date}",
            fontsize=15
        )

        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            filename.replace(
                "_masked.tif",
                "_preview.png"
            )
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("Saved:", output_path)


if __name__ == "__main__":

    files = sorted(
        glob.glob(
            os.path.join(INPUT_DIR, "*.tif")
        )
    )

    if not files:
        raise FileNotFoundError(
            "No masked satellite images found."
        )

    for filepath in files:
        create_previews(filepath)

    print("\nAll satellite previews generated.")
