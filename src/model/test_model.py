"""
Test CNN Model
"""

from change_detection_cnn import build_change_detection_model


def main():

    print("Building CNN model...\n")

    model = build_change_detection_model()

    model.summary()

    print("\n✅ CNN model created successfully.")


if __name__ == "__main__":
    main()