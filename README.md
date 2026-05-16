# Data Pipeline for Geothermal Plant Logs Analysis
**Course:** Computer Programming & Data Science  
**Reporting Standard:** IEEE Two-Column Research Format Integration  
**Student Name:** Lois Yvonne Trinidad  
**Student Number:** TUPM-25-0298  
**Assigned Topic ID:** RMS-04 (Geothermal Heat Exchanger Fouling)  
**Unique Filter Logic:** `fuel == 'Geothermal'` & `country == 'Philippines'`

---

## 🛠️ System Overview
This repository contains an automated, object-oriented software engineering pipeline designed to ingest, cleanse, analyze, and visualize live power plant generation telemetry. The operational target isolates Philippines-based geothermal energy assets to evaluate capacity thresholds and grid contribution distributions.

### Core Features
* **Memory-Optimized Ingestion:** Built using a low-memory chunk-streaming framework to process dataset modules without local engine lockups.
* **Automated Casing Mapping:** Programmed with automated metadata fallback sensors to map target engineering headers dynamically.
* **Vectorized Analytical Suite:** Powered strictly by NumPy to perform calculations for descriptive statistical vectors (Mean, Median, Std Dev, Variance, and Skewness).
* **Multi-Format Visual Engine:** Outputs both static data plots and dynamic `.gif` system animations.

---

## 📁 Repository Directory Structure
Per submission requirements, the workspace follows this strict architecture layout:
```text
EDS_TUPM-25-0298_Trinidad/
├── data/
│   ├── dataset_original.csv       # Sourced raw database logs
│   └── dataset_cleaned.csv        # Cleaned, unique database slice records
├── outputs/
│   ├── static_distribution_histogram.png   # Density distribution profile
│   ├── static_variability_boxplot.png       # Operational boundary limits
│   ├── static_correlation_scatter.png       # System capacity scaling map
│   ├── animated_system_trend.gif            # Real-time telemetry trend sweep
│   └── animated_cumulative_growth.gif       # Aggregate grid capacity growth profile
├── main.py                        # Complete data pipeline codebase
├── requirements.txt               # Library dependency ecosystem manifest
└── README.md                      # Pipeline system documentation