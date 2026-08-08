"""
Project Configuration File
"""

import os

# ==========================================================
# Project Information
# ==========================================================

PROJECT_NAME = "AI-Based Human-Induced Change Detection Using Satellite Images"

# ==========================================================
# Project Directories
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

LEVIR_DATASET = os.path.join(DATASET_DIR, "LEVIR-CD")

IMAGE_A_DIR = os.path.join(LEVIR_DATASET, "A")
IMAGE_B_DIR = os.path.join(LEVIR_DATASET, "B")
LABEL_DIR = os.path.join(LEVIR_DATASET, "label")

# ==========================================================
# Image Configuration
# ==========================================================

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# ==========================================================
# Model Configuration
# ==========================================================

MODEL_NAME = "CNN"

# ==========================================================
# Training Configuration
# ==========================================================

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001