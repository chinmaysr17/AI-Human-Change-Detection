"""
User Interface for AI-Based Human-Induced Change Detection

Allows the user to enter a LEVIR-CD image filename
and automatically runs the end-to-end detection pipeline.
"""

import os
import subprocess
import sys


# ============================================================
# MAIN USER INTERFACE
# ============================================================

def main():

    print("=" * 60)
    print("AI-BASED HUMAN-INDUCED CHANGE DETECTION")
    print("USER INTERFACE")
    print("=" * 60)

    print(
        "\nEnter the filename of a LEVIR-CD image pair."
    )

    print(
        "Example: test_1.png"
    )

    print(
        "The same filename must exist in A, B and label folders."
    )

    # --------------------------------------------------------
    # GET USER INPUT
    # --------------------------------------------------------

    filename = input(
        "\nEnter image filename: "
    ).strip()

    # --------------------------------------------------------
    # BASIC INPUT VALIDATION
    # --------------------------------------------------------

    if not filename:

        print(
            "\n❌ No filename entered."
        )

        return

    if os.path.basename(filename) != filename:

        print(
            "\n❌ Please enter only the filename."
        )

        print(
            "Example: test_1.png"
        )

        return

    # --------------------------------------------------------
    # CHECK FILE EXTENSION
    # --------------------------------------------------------

    allowed_extensions = (
        ".png",
        ".jpg",
        ".jpeg"
    )

    if not filename.lower().endswith(
        allowed_extensions
    ):

        print(
            "\n❌ Unsupported image format."
        )

        print(
            "Supported formats: PNG, JPG, JPEG"
        )

        return

    # --------------------------------------------------------
    # LOCATE MAIN PIPELINE
    # --------------------------------------------------------

    project_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    pipeline_path = os.path.join(
        project_dir,
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
        "Input:",
        filename
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

    print(
        "\n" + "=" * 60
    )

    print(
        "USER INTERFACE COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        "\n✅ Change detection completed successfully."
    )

    print(
        "Input image pair:",
        filename
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()