# AI-Based Human-Induced Change Detection Using Satellite Images

## Project Overview

This project develops an AI-based system for detecting human-induced changes from satellite image pairs using a Convolutional Neural Network (CNN).

The system takes two satellite images of the same geographical area captured at different times:

- **Image A** - Earlier image
- **Image B** - Later image

The two images are preprocessed and combined to form a **6-channel input**:

```text
Image A (3 channels)
        +
Image B (3 channels)
        ↓
6-Channel Input
        ↓
Final CNN Model
        ↓
Change Prediction
        ↓
Binary Change Map
        ↓
Change Visualization