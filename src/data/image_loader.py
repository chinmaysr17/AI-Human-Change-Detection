"""
LEVIR-CD Image Loader

Loads:
    Image A  -> Before image
    Image B  -> After image
    Label    -> Ground-truth change mask
"""

import cv2
import config


def load_image_pair(filename):
    """
    Load before image, after image, and ground-truth label.

    Returns:
        image_a, image_b, label

    Returns (None, None, None) if loading or validation fails.
    """

    # --------------------------------------------------------
    # BUILD FILE PATHS
    # --------------------------------------------------------

    image_a_path = config.IMAGE_A_DIR + "\\" + filename
    image_b_path = config.IMAGE_B_DIR + "\\" + filename
    label_path = config.LABEL_DIR + "\\" + filename

    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)
    label = cv2.imread(
        label_path,
        cv2.IMREAD_GRAYSCALE
    )

    # --------------------------------------------------------
    # CHECK IMAGE A
    # --------------------------------------------------------

    if image_a is None:

        print(
            f"❌ Unable to load Image A: "
            f"{image_a_path}"
        )

        return None, None, None

    # --------------------------------------------------------
    # CHECK IMAGE B
    # --------------------------------------------------------

    if image_b is None:

        print(
            f"❌ Unable to load Image B: "
            f"{image_b_path}"
        )

        return None, None, None

    # --------------------------------------------------------
    # CHECK LABEL
    # --------------------------------------------------------

    if label is None:

        print(
            f"❌ Unable to load Label: "
            f"{label_path}"
        )

        return None, None, None

    # --------------------------------------------------------
    # CHECK IMAGE DIMENSIONS
    # --------------------------------------------------------

    if image_a.shape[:2] != image_b.shape[:2]:

        print(
            "\n❌ Image dimension mismatch."
        )

        print(
            "Image A shape:",
            image_a.shape
        )

        print(
            "Image B shape:",
            image_b.shape
        )

        return None, None, None

    # --------------------------------------------------------
    # CHECK LABEL DIMENSIONS
    # --------------------------------------------------------

    if image_a.shape[:2] != label.shape[:2]:

        print(
            "\n❌ Label dimension mismatch."
        )

        print(
            "Image shape:",
            image_a.shape
        )

        print(
            "Label shape:",
            label.shape
        )

        return None, None, None

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    print(
        "✅ Image A loaded successfully."
    )

    print(
        "✅ Image B loaded successfully."
    )

    print(
        "✅ Label loaded successfully."
    )

    print(
        "✅ Image dimensions verified."
    )

    print(
        "Image dimensions:",
        image_a.shape[:2]
    )

    return image_a, image_b, label