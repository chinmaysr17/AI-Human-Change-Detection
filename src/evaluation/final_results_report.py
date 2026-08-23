"""
Final Project Results Report

Creates a concise text report containing:
- Baseline CNN results
- Improved CNN results
- Final held-out test results
- Comparison
- Project observations
"""

import os
import sys

# --------------------------------------------------
# Add src directory to Python path
# --------------------------------------------------

SRC_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, SRC_DIR)

import config


def main():

    print("=" * 60)
    print("FINAL PROJECT RESULTS REPORT")
    print("=" * 60)

    # --------------------------------------------------
    # Results obtained during project evaluation
    # --------------------------------------------------

    baseline = {
        "precision": 36.26,
        "recall": 61.84,
        "f1": 45.71,
        "iou": 29.63
    }

    improved = {
        "precision": 41.88,
        "recall": 70.47,
        "f1": 52.54,
        "iou": 35.63
    }

    final_test = {
        "precision": 29.48,
        "recall": 86.72,
        "f1": 44.00,
        "iou": 28.21
    }

    # --------------------------------------------------
    # Calculate improvements
    # --------------------------------------------------

    precision_change = (
        improved["precision"]
        - baseline["precision"]
    )

    recall_change = (
        improved["recall"]
        - baseline["recall"]
    )

    f1_change = (
        improved["f1"]
        - baseline["f1"]
    )

    iou_change = (
        improved["iou"]
        - baseline["iou"]
    )

    # --------------------------------------------------
    # Display Baseline Results
    # --------------------------------------------------

    print("\nBASELINE CNN")
    print("-" * 60)

    print(
        f"Precision : {baseline['precision']:.2f}%"
    )

    print(
        f"Recall    : {baseline['recall']:.2f}%"
    )

    print(
        f"F1-Score  : {baseline['f1']:.2f}%"
    )

    print(
        f"IoU       : {baseline['iou']:.2f}%"
    )

    # --------------------------------------------------
    # Display Improved Results
    # --------------------------------------------------

    print("\nIMPROVED CNN")
    print("-" * 60)

    print(
        f"Precision : {improved['precision']:.2f}%"
    )

    print(
        f"Recall    : {improved['recall']:.2f}%"
    )

    print(
        f"F1-Score  : {improved['f1']:.2f}%"
    )

    print(
        f"IoU       : {improved['iou']:.2f}%"
    )

    # --------------------------------------------------
    # Display Improvement
    # --------------------------------------------------

    print("\nIMPROVEMENT OVER BASELINE")
    print("-" * 60)

    print(
        f"Precision change : {precision_change:+.2f} percentage points"
    )

    print(
        f"Recall change    : {recall_change:+.2f} percentage points"
    )

    print(
        f"F1-Score change  : {f1_change:+.2f} percentage points"
    )

    print(
        f"IoU change       : {iou_change:+.2f} percentage points"
    )

    # --------------------------------------------------
    # Display Final Test Results
    # --------------------------------------------------

    print("\nFINAL HELD-OUT TEST SET")
    print("-" * 60)

    print(
        "Test samples : 13"
    )

    print(
        f"Precision    : {final_test['precision']:.2f}%"
    )

    print(
        f"Recall       : {final_test['recall']:.2f}%"
    )

    print(
        f"F1-Score     : {final_test['f1']:.2f}%"
    )

    print(
        f"IoU          : {final_test['iou']:.2f}%"
    )

    # --------------------------------------------------
    # Create final report text
    # --------------------------------------------------

    report = f"""
============================================================
AI-BASED HUMAN-INDUCED CHANGE DETECTION
FINAL PERFORMANCE REPORT
============================================================

DATASET
------------------------------------------------------------
Total samples       : 128
Training samples    : 102
Validation samples  : 13
Final test samples  : 13

MODEL
------------------------------------------------------------
Architecture        : CNN
Input               : 6-channel satellite image pair
Final model         : change_detection_cnn_final.keras

BASELINE CNN RESULTS
------------------------------------------------------------
Precision           : {baseline['precision']:.2f}%
Recall              : {baseline['recall']:.2f}%
F1-Score            : {baseline['f1']:.2f}%
IoU                 : {baseline['iou']:.2f}%

IMPROVED CNN RESULTS
------------------------------------------------------------
Precision           : {improved['precision']:.2f}%
Recall              : {improved['recall']:.2f}%
F1-Score            : {improved['f1']:.2f}%
IoU                 : {improved['iou']:.2f}%

IMPROVEMENT OVER BASELINE
------------------------------------------------------------
Precision           : {precision_change:+.2f} percentage points
Recall              : {recall_change:+.2f} percentage points
F1-Score            : {f1_change:+.2f} percentage points
IoU                 : {iou_change:+.2f} percentage points

FINAL HELD-OUT TEST RESULTS
------------------------------------------------------------
Test samples        : 13
Best threshold      : 0.50
Precision           : {final_test['precision']:.2f}%
Recall              : {final_test['recall']:.2f}%
F1-Score            : {final_test['f1']:.2f}%
IoU                 : {final_test['iou']:.2f}%

OBSERVATIONS
------------------------------------------------------------
1. The improved CNN performed better than the baseline CNN
   on the validation-set F1-score and IoU.

2. The improved CNN achieved:

   Precision : {improved['precision']:.2f}%
   Recall    : {improved['recall']:.2f}%
   F1-Score  : {improved['f1']:.2f}%
   IoU       : {improved['iou']:.2f}%

3. The final model was evaluated on 13 completely held-out
   test samples that were not used during training.

4. The final test recall of {final_test['recall']:.2f}% indicates
   that the model detected a large proportion of actual
   changed regions.

5. The final test precision of {final_test['precision']:.2f}%
   indicates that false-positive detections are still present.

6. Therefore, the current model demonstrates promising
   change-detection capability but still has scope for
   improving precision and reducing false positives.

VISUALIZATION
------------------------------------------------------------
13 final test visualizations were generated.

Location:
outputs/final_test_visualizations/

Each visualization contains:

Before Image
After Image
Ground Truth
Final CNN Prediction
Change Overlay

============================================================
END OF FINAL PERFORMANCE REPORT
============================================================
"""

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------

    output_dir = os.path.join(
        config.BASE_DIR,
        "outputs"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    report_path = os.path.join(
        output_dir,
        "final_project_results_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # --------------------------------------------------
    # Completion message
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("REPORT SAVED")
    print("=" * 60)

    print(
        "\nReport path:",
        report_path
    )

    print(
        "\n✅ Final project results report generated successfully."
    )


if __name__ == "__main__":
    main()