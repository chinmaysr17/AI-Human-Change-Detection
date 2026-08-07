"""
AI-Based Human-Induced Change Detection Using Satellite Images
Main Entry Point
"""

import config


def main():
    print("=" * 60)
    print(config.PROJECT_NAME)
    print("=" * 60)
    print(f"Dataset Folder : {config.DATASET_PATH}")
    print(f"Image Size     : {config.IMAGE_WIDTH} x {config.IMAGE_HEIGHT}")
    print(f"Model          : {config.MODEL_NAME}")
    print("=" * 60)
    print("Project setup completed successfully!")


if __name__ == "__main__":
    main()