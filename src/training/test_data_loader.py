"""
Test Training Data Loader
"""

from data_loader import load_dataset


def main():

    print("=" * 60)
    print("TRAINING DATA LOADER TEST")
    print("=" * 60)

    images, labels = load_dataset()

    print("\nFinal Dataset Information")
    print("----------------------------")
    print("Images:", images.shape)
    print("Labels:", labels.shape)

    print("\nImage data type:", images.dtype)
    print("Label data type:", labels.dtype)

    print("\nImage value range:")
    print(images.min(), "to", images.max())

    print("\nLabel values:")
    print(labels.min(), "to", labels.max())

    print("\n✅ Training data loader test completed.")


if __name__ == "__main__":
    main()