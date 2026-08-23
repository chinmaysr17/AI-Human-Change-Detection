"""
CNN Results Visualization
Baseline vs Improved CNN
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to Python path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import config


def main():

    print("=" * 60)
    print("CNN RESULTS VISUALIZATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Model performance results
    # --------------------------------------------------

    metrics = [
        "Precision",
        "Recall",
        "F1-Score",
        "IoU"
    ]

    baseline = np.array([
        36.26,
        61.84,
        45.71,
        29.63
    ])

    improved = np.array([
        41.88,
        70.47,
        52.54,
        35.63
    ])

    # --------------------------------------------------
    # 2. Create output directory
    # --------------------------------------------------

    output_dir = os.path.join(
        config.BASE_DIR,
        "outputs"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------
    # 3. Create comparison chart
    # --------------------------------------------------

    print("\nCreating comparison chart...")

    x = np.arange(len(metrics))

    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        baseline,
        width,
        label="Baseline CNN"
    )

    plt.bar(
        x + width / 2,
        improved,
        width,
        label="Improved CNN"
    )

    # --------------------------------------------------
    # 4. Chart labels
    # --------------------------------------------------

    plt.xlabel("Evaluation Metrics")

    plt.ylabel("Performance (%)")

    plt.title(
        "Baseline CNN vs Improved CNN"
    )

    plt.xticks(
        x,
        metrics
    )

    plt.ylim(
        0,
        100
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    # --------------------------------------------------
    # 5. Add values above bars
    # --------------------------------------------------

    for i in range(len(metrics)):

        plt.text(
            x[i] - width / 2,
            baseline[i] + 1,
            f"{baseline[i]:.2f}%",
            ha="center",
            fontsize=9
        )

        plt.text(
            x[i] + width / 2,
            improved[i] + 1,
            f"{improved[i]:.2f}%",
            ha="center",
            fontsize=9
        )

    plt.tight_layout()

    # --------------------------------------------------
    # 6. Save chart
    # --------------------------------------------------

    output_path = os.path.join(
        output_dir,
        "baseline_vs_improved_cnn.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\n✅ Comparison chart saved successfully."
    )

    print(
        "Output path:",
        output_path
    )

    # --------------------------------------------------
    # 7. Display chart
    # --------------------------------------------------

    plt.show()

    print(
        "\n✅ Results visualization completed successfully."
    )


if __name__ == "__main__":
    main()