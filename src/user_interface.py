"""
User Interface for AI-Based Human-Induced Change Detection

Allows the user to select a LEVIR-CD image pair by number
and automatically runs the end-to-end detection pipeline.
"""

import os
import re
import subprocess
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_A_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "LEVIR-CD",
    "A"
)


# ============================================================
# GET AVAILABLE IMAGE PAIRS
# ============================================================

def get_available_images():

    if not os.path.exists(DATASET_A_DIR):

        print(
            "\n❌ Dataset folder not found:"
        )

        print(
            DATASET_A_DIR
        )

        return []

    files = []

    for file in os.listdir(DATASET_A_DIR):

        if file.lower().endswith(".png"):

            if re.match(
                r"^test_\d+\.png$",
                file,
                re.IGNORECASE
            ):

                files.append(file)

    # Sort numerically instead of alphabetically
    files.sort(
        key=lambda name: int(
            re.search(
                r"test_(\d+)",
                name,
                re.IGNORECASE
            ).group(1)
        )
    )

    return files


# ============================================================
# MAIN USER INTERFACE
# ============================================================

def main():

    print("=" * 60)
    print("AI-BASED HUMAN-INDUCED CHANGE DETECTION")
    print("USER INTERFACE")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD AVAILABLE IMAGE PAIRS
    # --------------------------------------------------------

    images = get_available_images()

    if not images:

        print(
            "\n❌ No valid image pairs found."
        )

        return

    print(
        f"\nAvailable image pairs: {len(images)}"
    )

    print(
        "\nExamples:"
    )

    print(
        "1  ->",
        images[0]
    )

    if len(images) >= 2:

        print(
            "2  ->",
            images[1]
        )

    if len(images) >= 3:

        print(
            "3  ->",
            images[2]
        )

    print(
        f"\n{len(images)} image pairs are available."
    )

    print(
        "Select an image pair by entering its number."
    )

    # --------------------------------------------------------
    # GET USER SELECTION
    # --------------------------------------------------------

    user_input = input(
        "\nEnter image number (1-"
        + str(len(images))
        + "): "
    ).strip()

    # --------------------------------------------------------
    # CHECK NUMERIC INPUT
    # --------------------------------------------------------

    if not user_input.isdigit():

        print(
            "\n❌ Invalid input."
        )

        print(
            "Please enter a number."
        )

        return

    image_number = int(user_input)

    # --------------------------------------------------------
    # CHECK RANGE
    # --------------------------------------------------------

    if image_number < 1 or image_number > len(images):

        print(
            "\n❌ Image number out of range."
        )

        print(
            f"Please enter a number between "
            f"1 and {len(images)}."
        )

        return

    # --------------------------------------------------------
    # SELECT IMAGE
    # --------------------------------------------------------

    filename = images[
        image_number - 1
    ]

    print(
        "\nSelected image pair:",
        filename
    )

    # --------------------------------------------------------
    # LOCATE MAIN PIPELINE
    # --------------------------------------------------------

    pipeline_path = os.path.join(
        PROJECT_DIR,
        "src",
        "main_pipeline.py"
    )

    if not os.path.exists(pipeline_path):

        print(
            "\n❌ Main pipeline not found."
        )

        print(
            "Expected path:",
            pipeline_path
        )

        return

    # --------------------------------------------------------
    # RUN MAIN PIPELINE
    # --------------------------------------------------------

    print(
        "\nStarting change detection..."
    )

    print(
        "-" * 60
    )

    result = subprocess.run(
        [
            sys.executable,
            pipeline_path,
            filename
        ]
    )

    # --------------------------------------------------------
    # CHECK PIPELINE RESULT
    # --------------------------------------------------------

    if result.returncode != 0:

        print(
            "\n❌ Change detection pipeline failed."
        )

        return

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    output_path = os.path.join(
        PROJECT_DIR,
        "outputs",
        "pipeline",
        f"{os.path.splitext(filename)[0]}"
        "_pipeline_result.png"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "CHANGE DETECTION COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        "\nSelected image:",
        filename
    )

    print(
        "Result:",
        output_path
    )

    print(
        "\n✅ Image pair successfully processed."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()