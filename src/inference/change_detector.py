import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model


# =========================================================
# ADD SRC DIRECTORY TO PYTHON PATH
# =========================================================

SRC_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if SRC_DIR not in sys.path:
    sys.path.insert(
        0,
        SRC_DIR
    )


# =========================================================
# PROJECT IMPORTS
# =========================================================

import config

from data.image_loader import (
    load_image_pair
)

from preprocessing.image_loader import (
    preprocess_images
)


# =========================================================
# CONVERT IMAGE TO UINT8 RGB
# =========================================================

def convert_to_uint8_rgb(image):
    """
    Convert an image into uint8 RGB format.

    This keeps the feature extraction code
    consistent for different image data types.
    """

    image = np.asarray(image)

    # -----------------------------------------------------
    # Handle floating-point images
    # -----------------------------------------------------

    if np.issubdtype(
        image.dtype,
        np.floating
    ):

        if image.max() <= 1.0:

            image = (
                image * 255.0
            )

        image = np.clip(
            image,
            0,
            255
        ).astype(
            np.uint8
        )

    # -----------------------------------------------------
    # Handle other image types
    # -----------------------------------------------------

    elif image.dtype != np.uint8:

        image = np.clip(
            image,
            0,
            255
        ).astype(
            np.uint8
        )


    # -----------------------------------------------------
    # Convert grayscale to RGB
    # -----------------------------------------------------

    if image.ndim == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )


    # -----------------------------------------------------
    # Handle single-channel images
    # -----------------------------------------------------

    elif (
        image.ndim == 3
        and image.shape[-1] == 1
    ):

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )


    # -----------------------------------------------------
    # Keep only RGB channels
    # -----------------------------------------------------

    if (
        image.ndim == 3
        and image.shape[-1] > 3
    ):

        image = image[:, :, :3]


    return image


# =========================================================
# CALCULATE RGB MEAN
# =========================================================

def calculate_rgb_mean(
    image_crop
):
    """
    Calculate average R, G and B values.
    """

    mean_values = np.mean(
        image_crop,
        axis=(0, 1)
    )

    return {
        "r": round(
            float(mean_values[0]),
            2
        ),
        "g": round(
            float(mean_values[1]),
            2
        ),
        "b": round(
            float(mean_values[2]),
            2
        )
    }


# =========================================================
# CALCULATE RGB STANDARD DEVIATION
# =========================================================

def calculate_rgb_std(
    image_crop
):
    """
    Calculate RGB standard deviation.

    Higher variation can indicate more
    visual complexity within the region.
    """

    std_values = np.std(
        image_crop,
        axis=(0, 1)
    )

    return {
        "r": round(
            float(std_values[0]),
            2
        ),
        "g": round(
            float(std_values[1]),
            2
        ),
        "b": round(
            float(std_values[2]),
            2
        )
    }


# =========================================================
# CALCULATE TEXTURE
# =========================================================

def calculate_texture(
    image_crop
):
    """
    Calculate grayscale intensity standard deviation.

    This is used as a simple texture-related
    statistical feature.
    """

    gray = cv2.cvtColor(
        image_crop,
        cv2.COLOR_RGB2GRAY
    )

    texture_std = np.std(
        gray
    )

    return round(
        float(texture_std),
        2
    )


# =========================================================
# CALCULATE EDGE DENSITY
# =========================================================

def calculate_edge_density(
    image_crop
):
    """
    Calculate the percentage of pixels
    identified as edges using Canny detection.
    """

    gray = cv2.cvtColor(
        image_crop,
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_pixels = int(
        np.sum(
            edges > 0
        )
    )

    total_pixels = int(
        edges.size
    )

    if total_pixels == 0:

        return 0.0


    edge_density = (
        edge_pixels
        / total_pixels
    ) * 100


    return round(
        float(edge_density),
        2
    )


# =========================================================
# CALCULATE BEFORE / AFTER DIFFERENCE
# =========================================================

def calculate_image_difference(
    before_crop,
    after_crop
):
    """
    Calculate visual difference between
    the Before and After region crops.
    """

    # -----------------------------------------------------
    # Make sure both crops have same size
    # -----------------------------------------------------

    height = min(
        before_crop.shape[0],
        after_crop.shape[0]
    )

    width = min(
        before_crop.shape[1],
        after_crop.shape[1]
    )


    before_crop = (
        before_crop[
            :height,
            :width
        ]
    )

    after_crop = (
        after_crop[
            :height,
            :width
        ]
    )


    # -----------------------------------------------------
    # Absolute pixel difference
    # -----------------------------------------------------

    difference = cv2.absdiff(
        before_crop,
        after_crop
    )


    # -----------------------------------------------------
    # Mean absolute RGB difference
    # -----------------------------------------------------

    mean_difference = np.mean(
        difference
    )


    # -----------------------------------------------------
    # Convert to percentage of
    # maximum possible RGB difference
    # -----------------------------------------------------

    difference_percentage = (
        mean_difference
        / 255.0
    ) * 100


    return {
        "mean_absolute_difference": round(
            float(mean_difference),
            2
        ),

        "difference_percentage": round(
            float(difference_percentage),
            2
        )
    }


# =========================================================
# EXTRACT VISUAL FEATURES FOR A REGION
# =========================================================

def extract_visual_features(
    before_image,
    after_image,
    region
):
    """
    Extract visual features from the same
    spatial region in Before and After images.

    Region coordinates are generated at
    256 × 256 model resolution.

    Original satellite images may have
    a larger resolution, so coordinates
    are scaled accordingly.
    """

    before_image = convert_to_uint8_rgb(
        before_image
    )

    after_image = convert_to_uint8_rgb(
        after_image
    )


    # -----------------------------------------------------
    # Model resolution
    # -----------------------------------------------------

    model_width = 256
    model_height = 256


    # -----------------------------------------------------
    # Original image dimensions
    # -----------------------------------------------------

    original_height = (
        before_image.shape[0]
    )

    original_width = (
        before_image.shape[1]
    )


    # -----------------------------------------------------
    # Scaling factors
    # -----------------------------------------------------

    scale_x = (
        original_width
        / model_width
    )

    scale_y = (
        original_height
        / model_height
    )


    # -----------------------------------------------------
    # Region bounding box
    # -----------------------------------------------------

    box = region[
        "bounding_box"
    ]


    x = int(
        round(
            box["x"] * scale_x
        )
    )

    y = int(
        round(
            box["y"] * scale_y
        )
    )

    width = int(
        round(
            box["width"] * scale_x
        )
    )

    height = int(
        round(
            box["height"] * scale_y
        )
    )


    # -----------------------------------------------------
    # Keep coordinates inside image
    # -----------------------------------------------------

    x = max(
        0,
        min(
            x,
            original_width - 1
        )
    )

    y = max(
        0,
        min(
            y,
            original_height - 1
        )
    )


    width = max(
        1,
        min(
            width,
            original_width - x
        )
    )

    height = max(
        1,
        min(
            height,
            original_height - y
        )
    )


    # -----------------------------------------------------
    # Crop corresponding Before region
    # -----------------------------------------------------

    before_crop = before_image[
        y:y + height,
        x:x + width
    ]


    # -----------------------------------------------------
    # Crop corresponding After region
    # -----------------------------------------------------

    after_crop = after_image[
        y:y + height,
        x:x + width
    ]


    # -----------------------------------------------------
    # Calculate features
    # -----------------------------------------------------

    before_rgb_mean = (
        calculate_rgb_mean(
            before_crop
        )
    )

    after_rgb_mean = (
        calculate_rgb_mean(
            after_crop
        )
    )


    before_rgb_std = (
        calculate_rgb_std(
            before_crop
        )
    )

    after_rgb_std = (
        calculate_rgb_std(
            after_crop
        )
    )


    before_texture = (
        calculate_texture(
            before_crop
        )
    )

    after_texture = (
        calculate_texture(
            after_crop
        )
    )


    before_edge_density = (
        calculate_edge_density(
            before_crop
        )
    )

    after_edge_density = (
        calculate_edge_density(
            after_crop
        )
    )


    image_difference = (
        calculate_image_difference(
            before_crop,
            after_crop
        )
    )


    # -----------------------------------------------------
    # Return visual features
    # -----------------------------------------------------

    return {

        "original_crop": {

            "x": x,

            "y": y,

            "width": width,

            "height": height
        },

        "before_rgb_mean":
            before_rgb_mean,

        "after_rgb_mean":
            after_rgb_mean,

        "before_rgb_std":
            before_rgb_std,

        "after_rgb_std":
            after_rgb_std,

        "before_texture_std":
            before_texture,

        "after_texture_std":
            after_texture,

        "before_edge_density_percent":
            before_edge_density,

        "after_edge_density_percent":
            after_edge_density,

        "before_after_difference":
            image_difference
    }


# =========================================================
# EXTRACT CHANGE REGIONS
# =========================================================

def extract_change_regions(
    change_mask,
    before_image,
    after_image,
    minimum_area=20
):
    """
    Extract individual connected change regions
    and calculate geometric + visual features.
    """

    # -----------------------------------------------------
    # Convert binary mask to 8-bit
    # -----------------------------------------------------

    binary_mask = (
        np.asarray(
            change_mask,
            dtype=np.uint8
        ) * 255
    )


    # -----------------------------------------------------
    # Connected components
    # -----------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )


    regions = []


    # -----------------------------------------------------
    # Total image pixels
    # -----------------------------------------------------

    total_image_pixels = int(
        change_mask.size
    )


    # -----------------------------------------------------
    # Process each component
    # -----------------------------------------------------

    for label_id in range(
        1,
        num_labels
    ):

        # -------------------------------------------------
        # Region area
        # -------------------------------------------------

        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA
            ]
        )


        # -------------------------------------------------
        # Remove tiny regions
        # -------------------------------------------------

        if area < minimum_area:
            continue


        # -------------------------------------------------
        # Bounding box
        # -------------------------------------------------

        x = int(
            stats[
                label_id,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                label_id,
                cv2.CC_STAT_TOP
            ]
        )

        width = int(
            stats[
                label_id,
                cv2.CC_STAT_WIDTH
            ]
        )

        height = int(
            stats[
                label_id,
                cv2.CC_STAT_HEIGHT
            ]
        )


        # -------------------------------------------------
        # Center
        # -------------------------------------------------

        center_x = float(
            centroids[
                label_id
            ][0]
        )

        center_y = float(
            centroids[
                label_id
            ][1]
        )


        # -------------------------------------------------
        # Area percentage
        # -------------------------------------------------

        area_percentage = (
            area
            / total_image_pixels
        ) * 100


        # -------------------------------------------------
        # Bounding box area
        # -------------------------------------------------

        bounding_box_area = (
            width * height
        )


        # -------------------------------------------------
        # Aspect ratio
        # -------------------------------------------------

        if height > 0:

            aspect_ratio = (
                width / height
            )

        else:

            aspect_ratio = 0.0


        # -------------------------------------------------
        # Fill ratio
        # -------------------------------------------------

        if bounding_box_area > 0:

            fill_ratio = (
                area
                / bounding_box_area
            ) * 100

        else:

            fill_ratio = 0.0


        # -------------------------------------------------
        # Basic region information
        # -------------------------------------------------

        region = {

            "area_pixels":
                area,

            "area_percentage":
                round(
                    area_percentage,
                    4
                ),

            "bounding_box": {

                "x":
                    x,

                "y":
                    y,

                "width":
                    width,

                "height":
                    height
            },

            "bounding_box_area":
                bounding_box_area,

            "aspect_ratio":
                round(
                    aspect_ratio,
                    3
                ),

            "fill_ratio":
                round(
                    fill_ratio,
                    2
                ),

            "center": {

                "x":
                    round(
                        center_x,
                        2
                    ),

                "y":
                    round(
                        center_y,
                        2
                    )
            }
        }


        # -------------------------------------------------
        # Add visual features
        # -------------------------------------------------

        visual_features = (
            extract_visual_features(
                before_image,
                after_image,
                region
            )
        )


        region[
            "visual_features"
        ] = visual_features


        regions.append(
            region
        )


    # =====================================================
    # SORT LARGEST TO SMALLEST
    # =====================================================

    regions.sort(
        key=lambda region:
            region["area_pixels"],
        reverse=True
    )


    # =====================================================
    # ASSIGN CLEAN REGION IDs
    # =====================================================

    for index, region in enumerate(
        regions,
        start=1
    ):

        region[
            "region_id"
        ] = index


    return regions


# =========================================================
# MAIN CHANGE DETECTION
# =========================================================

def run_change_detection(
    filename
):

    filename = filename.strip()


    # =====================================================
    # LOAD IMAGE PAIR
    # =====================================================

    (
        image_a,
        image_b,
        label
    ) = load_image_pair(
        filename
    )


    if (
        image_a is None
        or image_b is None
        or label is None
    ):

        raise ValueError(
            "Image pair could not be loaded."
        )


    # =====================================================
    # VALIDATE DIMENSIONS
    # =====================================================

    if (
        image_a.shape[:2]
        != image_b.shape[:2]
    ):

        raise ValueError(
            "Image A and Image B dimensions "
            "do not match."
        )


    if (
        image_a.shape[:2]
        != label.shape[:2]
    ):

        raise ValueError(
            "Image and label dimensions "
            "do not match."
        )


    # =====================================================
    # PREPROCESS IMAGES
    # =====================================================

    (
        processed_a,
        processed_b,
        processed_label
    ) = preprocess_images(
        image_a,
        image_b,
        label
    )


    # =====================================================
    # COMBINE BEFORE + AFTER
    # =====================================================

    combined_image = np.concatenate(
        [
            processed_a,
            processed_b
        ],
        axis=-1
    )


    if (
        combined_image.shape[-1]
        != 6
    ):

        raise ValueError(
            "Expected 6-channel input."
        )


    # =====================================================
    # ADD BATCH DIMENSION
    # =====================================================

    model_input = np.expand_dims(
        combined_image,
        axis=0
    )


    # =====================================================
    # MODEL PATH
    # =====================================================

    model_path = os.path.join(
        config.BASE_DIR,
        "models",
        "change_detection_cnn_final.keras"
    )


    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )


    # =====================================================
    # LOAD CNN
    # =====================================================

    model = load_model(
        model_path,
        compile=False
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = model.predict(
        model_input,
        verbose=0
    )[0]


    # =====================================================
    # THRESHOLD
    # =====================================================

    threshold = 0.50


    # =====================================================
    # BINARY CHANGE MASK
    # =====================================================

    predicted_change = (
        prediction >= threshold
    ).astype(
        np.uint8
    )


    predicted_change = (
        predicted_change.squeeze()
    )


    # =====================================================
    # CHANGED PIXELS
    # =====================================================

    predicted_pixels = int(
        np.sum(
            predicted_change == 1
        )
    )


    total_pixels = int(
        predicted_change.size
    )


    # =====================================================
    # CHANGE PERCENTAGE
    # =====================================================

    change_percentage = (
        predicted_pixels
        / total_pixels
    ) * 100


    # =====================================================
    # EXTRACT REGIONS
    # =====================================================

    change_regions = (
        extract_change_regions(
            predicted_change,
            image_a,
            image_b,
            minimum_area=20
        )
    )


    # =====================================================
    # RETURN RESULT
    # =====================================================

    return {

        "filename":
            filename,

        "threshold":
            threshold,

        "prediction_shape":
            prediction.shape,

        "predicted_change_pixels":
            predicted_pixels,

        "total_pixels":
            total_pixels,

        "change_percentage":
            change_percentage,

        "prediction_min":
            float(
                prediction.min()
            ),

        "prediction_max":
            float(
                prediction.max()
            ),

        "prediction_mean":
            float(
                prediction.mean()
            ),

        "change_map":
            predicted_change,

        "change_regions":
            change_regions
    }