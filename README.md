# AI-Based Human-Induced Change Detection Using Satellite Images

## 1. Project Overview

AI-Based Human-Induced Change Detection Using Satellite Images is a satellite image analysis project that explores the detection and visualization of changes in geographical areas using image processing, deep learning, and Sentinel-2 satellite imagery.

The system compares satellite images acquired at different time periods to identify potential surface changes. It combines a CNN-based binary change-detection model with a Sentinel-2 image acquisition and preprocessing workflow.

The project also explores mining-related land-change analysis using satellite imagery and experimental segmentation models.

## 2. Objectives

- Acquire satellite imagery for a selected geographical area and time period.
- Preprocess and prepare satellite images for comparison.
- Develop a CNN-based image change-detection model.
- Generate pixel-level change probability maps and binary change maps.
- Visualize before-and-after satellite imagery and potential changes.
- Explore mining-related land disturbance analysis.
- Develop a web application for location-based satellite image analysis.

## 3. System Architecture

The overall workflow is:

```text
        User Location and Date Selection
                    |
                    v
          Sentinel-2 Image Retrieval
                    |
                    v
             Image Preprocessing
                    |
                    v
          Before and After Images
                    |
                    v
             CNN Change Detection
                    |
                    v
           Change Probability Map
                    |
                    v
             Binary Change Map
                    |
                    v
          Results and Visualization
```

The Sentinel-2 acquisition and preprocessing workflow and the LEVIR-CD-based CNN workflow are separate components. Their results should not be interpreted as a single independently validated system.

## 4. Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Mapping | Leaflet.js |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV |
| Geospatial Processing | Rasterio |
| Satellite Imagery | Sentinel-2 Level-2A |
| Model Dataset | LEVIR-CD |
| Version Control | Git and GitHub |

## 5. Satellite Image Acquisition

The project uses Sentinel-2 Level-2A satellite imagery.

The acquisition workflow supports:
- Geographical location selection.
- Latitude and longitude selection.
- Analysis radius selection.
- Before and after date selection.
- Satellite image retrieval.
- Acquisition metadata storage.

### Demonstration acquisition

One successfully retrieved demonstration is:

| Parameter | Before | After |
|---|---|---|
| Acquisition date | 21 January 2024 | 20 January 2026 |
| Satellite | Sentinel-2A | Sentinel-2C |
| Cloud coverage | 0.97% | 0% |
| Product level | Level-2A | Level-2A |

Study area: Shivamogga, Karnataka, India.

## 6. Image Preprocessing

The Sentinel-2 preprocessing workflow includes:

1. Satellite band extraction.
2. Spectral band selection.
3. Image alignment and grid preparation.
4. Surface reflectance calibration using product metadata.
5. Scene Classification Layer (SCL)-based cloud and invalid-pixel masking.
6. Image cropping and preparation.
7. Before-and-after image stacking.

### Spectral bands

The model-ready experimental Sentinel-2 raster contains eight bands:

| Band | Description |
|---|---|
| B02 | Blue |
| B03 | Green |
| B04 | Red |
| B08 | Near Infrared |

The four bands are stored for both acquisition dates, producing an eight-band raster.

## 7. CNN-Based Change Detection

The project includes a CNN encoder-decoder architecture for binary satellite image change detection.

### Model architecture

- Convolutional encoder.
- Max-pooling layers.
- Feature extraction layers.
- Upsampling layers.
- Final sigmoid activation layer.

### Model configuration

| Parameter | Value |
|---|---:|
| Architecture | CNN Encoder-Decoder |
| Input size | 256 × 256 × 6 |
| Output size | 256 × 256 × 1 |
| Parameters | 167,969 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 8 |
| Epochs | 20 |
| Loss function | Weighted Binary Cross-Entropy |
| Positive class weight | 3.0 |
| Prediction threshold | 0.50 |

The model receives two RGB images concatenated along the channel dimension, resulting in a six-channel input.

The sigmoid output represents pixel-level change probabilities. Applying the selected threshold produces a binary change mask.

## 8. Dataset and Model Evaluation

The CNN was trained and evaluated using the LEVIR-CD dataset.

### Dataset split

| Dataset | Samples |
|---|---:|
| Training | 102 |
| Validation | 13 |
| Testing | 13 |
| Total | 128 |

The split uses random state 42.

### Final model evaluation

| Metric | Result |
|---|---:|
| Precision | 29.48% |
| Recall | 86.72% |
| F1-score | 44.00% |
| IoU | 28.21% |

### Model comparison

Three CNN model versions were compared using the same held-out LEVIR-CD test split.

| Model | Precision | Recall | F1-score | IoU |
|---|---:|---:|---:|---:|
| Original | 0% | 0% | 0% | 0% |
| Final | 29.48% | 86.72% | 44.00% | 28.21% |
| Improved | 43.06% | 75.08% | 54.73% | 37.67% |

The Improved model achieved the highest F1-score and IoU in this comparison.

**Evaluation limitation:** The test set contains only 13 images. These results should be interpreted as preliminary evaluation results rather than broad evidence of generalization.

## 9. Sentinel-2 Model Testing

The existing LEVIR-CD-trained CNN was experimentally tested using Sentinel-2 imagery.

The model uses six RGB channels, while the separate Sentinel-2 model-ready preprocessing workflow produces eight spectral bands. Therefore, the current CNN does not directly consume the eight-band model-ready raster.

For the demonstration acquisition, neither the Final nor Improved model generated pixels above the production threshold of 0.50.

Experimental testing also identified strong prediction sensitivity near image boundaries.

These findings indicate that the existing CNN requires further domain-specific training and validation before it can be considered reliable for Sentinel-2 change detection.

## 10. Temporal Spectral Analysis

A separate Sentinel-2 investigation examined unusual reflectance values at three locations in the September 2026 and October 2026 acquisitions.

The investigation included:

- Comparing original JP2 values with processed raster values.
- Checking reflectance calibration against product metadata.
- Inspecting cloud and scene classification masks.
- Examining local spectral neighbourhoods.
- Comparing before-and-after reflectance.
- Calculating NDVI changes.

Three investigated locations showed different NDVI responses:

| Location | Before NDVI | After NDVI | Change |
|---|---:|---:|---:|
| September bright feature | 0.1074 | 0.2452 | +0.1378 |
| October B02 anomaly | 0.2151 | 0.2415 | +0.0263 |
| October bright feature | 0.2921 | 0.1564 | -0.1357 |

These observations are spectral-change candidates. They have not been independently confirmed as human-induced changes.

## 11. Mining-Related Change Analysis

The project also contains an experimental mining-related image analysis workflow.

The experimental work includes:

- Mining image and mask inspection.
- Dataset quality validation.
- Polygon overlap checks.
- Dataset split preparation.
- Mining-specific data loaders.
- U-Net-based segmentation experiments.
- Model comparison and validation scripts.

The mining dataset and experimental models are separate from the LEVIR-CD CNN workflow.

Mining dataset validation identified 28 images requiring review in one validation run. These samples should not be treated as fully validated training data without completing the required quality checks.

## 12. Web Application

The project includes a browser-based interface connected to a FastAPI backend.

### Frontend features

- Location search.
- Interactive geographical map.
- Latitude and longitude selection.
- Analysis radius selection.
- Before-and-after date selection.
- Model information display.
- Analysis history.

### Backend features

The FastAPI application exposes the following endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Backend health check |
| `/api/model` | GET | Model information |
| `/api/analysis-selection` | POST | Save analysis selection |
| `/api/analysis-history` | GET | Retrieve analysis history |
| `/api/analyze-location` | POST | Location-based Sentinel-2 analysis |
| `/api/analyze/{filename}` | GET | Existing dataset-image analysis |

The location-based analysis workflow retrieves satellite imagery, prepares the image pair, performs experimental prediction, and stores the analysis results.

## 13. Project Structure

```text
AI-Human-Change-Detection/
│
├── backend/
│   └── app/
│       └── main.py
│
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
│       ├── api.js
│       ├── map.js
│       ├── analysis.js
│       ├── history.js
│       └── main.js
│
├── data/
│   ├── sentinel2/
│   │   ├── raw/
│   │   ├── processed/
│   │   ├── analyses/
│   │   └── mining/
│   └── LEVIR-CD/
│
├── models/
│
├── reports/
│
├── src/
│   ├── config.py
│   ├── data/
│   ├── inference/
│   ├── model/
│   ├── preprocessing/
│   └── training/
│
├── requirements.txt
├── README.md
└── .gitignore
```

This is a simplified overview of the main project directories.

## 14. Installation and Setup

### Prerequisites

- Python 3.10 or later, compatible with the installed project dependencies.
- Git.
- VS Code or another Python development environment.

### Clone the repository

```bash
git clone https://github.com/chinmaysr17/AI-Human-Change-Detection.git

cd AI-Human-Change-Detection
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Copy the example configuration into a project-root `.env` file.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Update the placeholders in `.env` with your own values:

| Variable | Purpose |
|---|---|
| `DB_HOST` | PostgreSQL server host |
| `DB_PORT` | PostgreSQL server port (usually `5432`) |
| `DB_NAME` | Existing project database name |
| `DB_USER` | PostgreSQL username |
| `DB_PASSWORD` | PostgreSQL password |
| `COPERNICUS_CLIENT_ID` | Copernicus Data Space client ID |
| `COPERNICUS_CLIENT_SECRET` | Copernicus Data Space client secret |
| `FRONTEND_ORIGIN` | Frontend origin (default `http://127.0.0.1:5501`) |
| `COOKIE_SECURE` | Set to `true` when using HTTPS; keep `false` for local HTTP |

The PostgreSQL database named by `DB_NAME` must already exist. Keep your real `.env` file private; never commit credentials or passwords to GitHub.

## 15. Running the Application

### Start the backend

```bash
python -m uvicorn backend.app.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### Start the frontend

Open a separate terminal:

```bash
python -m http.server 5501 --directory frontend
```

Frontend URL:

```text
http://127.0.0.1:5501/index.html
```

Keep both servers running while using the application.

## 16. Current Limitations

- The existing CNN is trained on LEVIR-CD rather than a dedicated Sentinel-2 dataset.
- The CNN has not been independently validated for Sentinel-2 imagery.
- The model performs binary change detection rather than multi-category human-induced change classification.
- The LEVIR-CD test split is small.
- Sentinel-2 spectral-change candidates require independent verification.
- Mining dataset samples require further quality validation.
- Boundary-sensitive model predictions require additional investigation.
- The current demonstration should be treated as an experimental workflow, not a fully validated operational monitoring system.

## 17. Future Enhancements

- Develop a Sentinel-2-specific training dataset with reliable reference labels.
- Train a multispectral change-detection model using Sentinel-2 bands.
- Improve image alignment and temporal consistency checks.
- Investigate boundary-aware inference strategies.
- Improve mining-related segmentation and dataset quality.
- Introduce multi-category human-induced change classification.
- Evaluate models using larger independent test datasets.
- Improve the web application's result visualization and reporting features.

## 18. Conclusion

This project establishes an experimental framework for satellite-based change analysis using image processing, deep learning, and geospatial technologies.

It integrates Sentinel-2 image retrieval, preprocessing, a CNN-based binary change-detection workflow, and a web application for location-based analysis.

The project also explores mining-related land-change analysis and multispectral temporal comparisons.

The current results demonstrate the implementation of the major workflows and identify important areas requiring further validation. Developing a dedicated Sentinel-2 training dataset and evaluating the model on independently labelled satellite imagery are the key next steps toward reliable human-induced change detection.

---

**Project:** AI-Based Human-Induced Change Detection Using Satellite Images

**Repository:** https://github.com/chinmaysr17/AI-Human-Change-Detection
## 11. Sentinel-2 Long-Term Change Analysis (January 2016–January 2026)

A separate temporal analysis was conducted using Sentinel-2 Level-2A imagery to investigate spectral and land-cover changes in Shivamogga, Karnataka, India, over approximately ten years.

### Study Area and Acquisition

- Study area: Shivamogga, Karnataka, India.
- Coordinate reference system: EPSG:32643 (UTM Zone 43N).
- Analysis extent: 10 km × 10 km.
- Raster resolution: 10 metres.
- Raster dimensions: 1000 × 1000 pixels.
- Satellite data: Sentinel-2 Level-2A.

| Parameter | Before | After |
|---|---|---|
| Target date | 20 January 2016 | 20 January 2026 |
| Selected acquisition | 13 January 2016 | 20 January 2026 |
| Satellite | Sentinel-2A | Sentinel-2C |
| Cloud coverage | 4.69% | 0% |
| Tile | T43PER | T43PER |
| Orbit | R062 | R062 |

Sentinel-2 imagery was used instead of January 2015 imagery because Sentinel-2A was launched in June 2015.

### Image Alignment and Quality Checks

Both images were prepared on a common projected grid with identical dimensions, bounds and 10-metre pixel resolution.

The Scene Classification Layer (SCL) was inspected for cloud, shadow and invalid-pixel classes. No cloud or shadow classes were identified within the study area in either selected image.

### Spectral Difference Analysis

A pixel-wise spectral difference analysis was performed using the aligned Blue, Green and Red bands.

| Metric | Result |
|---|---:|
| Common valid pixels | 1,000,000 (100%) |
| Mean absolute difference | 0.023230 |
| Median absolute difference | 0.016067 |
| 95th percentile | 0.066933 |
| Maximum difference | 0.921000 |

### Candidate Change Detection

Two percentile-based thresholds were used to identify pixels with relatively high spectral differences.

| Threshold | Candidate pixels | Area |
|---|---:|---:|
| 95th percentile | 50,023 | 500.23 ha |
| 98th percentile | 20,011 | 200.11 ha |

Spatial refinement was performed using a 3 × 3 median filter, 8-connected component analysis and removal of regions smaller than 9 pixels.

| Threshold | Refined pixels | Refined area | Regions |
|---|---:|---:|---:|
| 95th percentile | 28,692 | 286.92 ha | 697 |
| 98th percentile | 9,190 | 91.90 ha | 282 |

These regions represent spectral-change candidates, not confirmed human-induced changes.

### Top Candidate Region Inspection

The ten largest candidate regions were extracted for further visual inspection.

The largest region contained 645 pixels, corresponding to 6.45 hectares.

One candidate region was manually inspected using historical Google Earth imagery:

- Coordinates: 13.915181° N, 75.556568° E.
- Historical imagery: 31 December 2016.
- Later imagery: 12 September 2026.
- Visual observation: vegetation in the earlier imagery and a substantially cleared or exposed-earth area with visible earthworks and access tracks in the later imagery.

This provides visual evidence of land-cover change at the inspected location. However, the cause has not been independently established, and the remaining nine regions have not been individually verified.

### Output Files

The analysis scripts generate outputs under:

`data/sentinel2/analyses/`

The main output directories are:

- `shivamogga_20160113_20260120_spectral/`
- `shivamogga_20160113_20260120_candidates/`
- `shivamogga_20160113_20260120_spatial_refinement/`
- `shivamogga_20160113_20260120_top10_regions/`

The workflow scripts include:

- `sentinel2_spectral_change_jan2016_2026.py`
- `create_sentinel2_candidate_masks.py`
- `refine_sentinel2_candidate_regions.py`
- `extract_top10_sentinel2