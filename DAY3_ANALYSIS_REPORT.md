\# Day 3 — Sentinel-2 Change Analysis Report



\## 1. Objective



To compare Sentinel-2 satellite imagery acquired on 27 September 2026 and 27 October 2026 over the Shivamogga study area and identify potential landscape changes using spectral difference analysis and NDVI.



\## 2. Study Area and Satellite Data



\- Study area: Shivamogga, Karnataka

\- Area: Approximately 10 km × 10 km

\- Satellite: Sentinel-2C

\- Product: Level-2A

\- Tile: T43PER

\- Before image: 27 September 2026

\- After image: 27 October 2026

\- Spatial resolution: 10 metres

\- Spectral bands: Blue, Green, Red and NIR



\## 3. Image Alignment and Valid Pixel Analysis



Both satellite images were verified for matching dimensions, CRS, pixel transform and geographic bounds.



\- Image dimensions: 1000 × 1000

\- CRS: EPSG:32643

\- Resolution: 10 metres

\- Common valid pixels: 991,518

\- Common valid area: 99.15%



\## 4. Spectral Difference Analysis



Absolute spectral differences were calculated across four bands.



Mean absolute spectral difference: 125.17 DN.



Median difference: 99.50 DN.



95th percentile: 295.75 DN.



The 98th percentile was used as an exploratory threshold to identify pixels with relatively high spectral differences.



\## 5. Candidate Change Detection



A preliminary candidate-change map was generated using the 98th percentile threshold.



\- Threshold: 403.75 DN

\- Candidate pixels before filtering: 19,855

\- Candidate pixels after filtering: 16,148

\- Connected regions: 942

\- Candidate area: 161.48 hectares

\- Candidate area percentage: 1.63%



Small connected regions containing fewer than four pixels were removed from the final candidate map.



\## 6. NDVI Change Analysis



NDVI was calculated for both dates using the Red and NIR bands.



NDVI = (NIR - Red) / (NIR + Red)



Results:



\- Mean NDVI change: +0.0020

\- Median absolute NDVI change: 0.0262

\- 90th percentile absolute NDVI change: 0.0815

\- High vegetation-change pixels: 99,152

\- Candidate pixels overlapping high vegetation-change pixels: 11,711



The results indicate that vegetation-index variation is associated with a substantial portion of the candidate detections.



\## 7. Exploratory Change Group Classification



Candidate pixels were separated using an absolute NDVI change threshold of 0.0815.



| Group | Pixels | Area |

|---|---:|---:|

| Vegetation-associated spectral changes | 11,709 | 117.09 ha |

| Other spectral changes | 4,439 | 44.39 ha |

| Total | 16,148 | 161.48 ha |



These groups represent spectral characteristics and are not confirmed land-cover categories.



\## 8. Region-Level Analysis



A region-level validation report was generated for 942 connected candidate regions.



Important regions:



| Region | Pixels | Area | Mean NDVI Change |

|---|---:|---:|---:|

| 842 | 707 | 7.07 ha | +0.20377 |

| 818 | 690 | 6.90 ha | +0.20412 |

| 335 | 520 | 5.20 ha | +0.07139 |

| 873 | 227 | 2.27 ha | +0.26116 |

| 39 | 217 | 2.17 ha | -0.12184 |

| 871 | 171 | 1.71 ha | -0.27990 |



Detailed visual comparisons were generated for selected regions.



Region 39 showed no obvious large-scale transformation in the supplied image comparison.



Region 842 showed substantial positive NDVI variation.



Region 871 showed negative NDVI variation, potentially associated with vegetation reduction.



These observations require further validation.



\## 9. Limitations



1\. Spectral differences may result from seasonal, atmospheric, illumination and surface-moisture variations.

2\. NDVI variation does not independently prove deforestation or agricultural activity.

3\. The current method does not reliably distinguish new buildings from existing structures.

4\. Candidate regions have not yet been validated against independent ground-truth data.

5\. The current classification is exploratory and does not identify all five intended human-induced change categories.

6\. The existing LEVIR-CD CNN is a building-change baseline and is not a validated general-purpose Sentinel-2 multispectral model.



\## 10. Conclusion



The project successfully acquired, aligned and processed Sentinel-2 imagery for two dates over Shivamogga.



Spectral difference analysis, NDVI comparison, exploratory candidate detection, change grouping and region-level analysis were implemented.



The resulting pipeline identifies potential landscape-change locations covering 161.48 hectares.



Further model development, category-specific datasets and ground-truth validation are required before these candidate changes can be classified as confirmed human-induced landscape changes.



\## 11. Day 3 Outcome



The preliminary Sentinel-2 comparison and candidate-change analysis pipeline has been implemented successfully. The outputs provide a foundation for more advanced change detection and category-specific classification.



