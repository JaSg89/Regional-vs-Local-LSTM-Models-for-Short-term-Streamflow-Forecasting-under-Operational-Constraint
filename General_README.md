# 📄 Regional vs. Local LSTM Models for Short-term Streamflow Forecasting under Operational Constraints

**Authors:** Saavedra-Garrido, J., Arévalo, J., De La Fuente, L., Tapia, A., Paredes-Arroyo, Córdova, A. M., Velandia, D., C., Álvarez, P., Reyes-Serrano, H., & Salas, R. (2025)

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)  ![Stars](https://img.shields.io/github/stars/your-username/Regional-vs-Local-LSTM-Models-for-Short-term-Streamflow-Forecasting-under-Operational-Constraint.svg?style=social)

This repository brings together all code, scripts, and resources to run and analyze both the **regional LSTM model** and **local LSTM models** for short-term streamflow forecasting under operational constraints.

---

## 📂 Repository Structure

- `Imputation/` — Scripts and classes for **operational imputation** of hydrological, climatic, and geographic variables.  
- `LOCAL_MODELS/` — Sequence generation and forecasting with local LSTM models, each trained independently for individual catchments.  
- `REGIONAL_MODELS/` — Data preparation and forecasting with the regional LSTM model, trained globally on data from all catchments.  
- `FIGURES/` — Scripts to generate figures, plots, and maps used in the article.  


---

## 📝 Overview

This research compares two LSTM training approaches:

1. **Regional Model:** Trained once on combined data from multiple catchments to learn global hydrological patterns.  
2. **Local Models:** Trained individually for each catchment, serving as a traditional benchmark.  

The goal is to evaluate which configuration improves the prediction of mean and maximum streamflow at 1- to 5-day horizons, captures extreme events, and generalizes under real operational conditions.

---

## 🚀 General Instructions

1. Clone the repository and create an environment (Conda or virtualenv) with the dependencies specified in each subfolder.  
2. First run the `Imputation/` scripts to generate complete, up-to-date time series.  
3. Then generate sequences and forecasts with `LOCAL_MODELS/` and evaluate the results.  
4. Finally, train and forecast with the global model using `REGIONAL_MODELS/`.  
5. Output results and metrics are stored in each module's `Trained_Models_TEST/` folder.  

---

## 🛠️ Common Dependencies

- Python 3.8+  
- pandas, numpy, scipy, h5py, scikit-learn, tensorflow (and keras), statsmodels, CRPS  
- Other standard libraries: `pickle`, `logging`, `os`, `time`, etc.  

Exact versions are detailed in each subfolder's README.

---

## How to cite

The associated article is currently under review.

**BibTeX:**
```bibtex
@unpublished{Autor2025_HidroCL,
  title        = {Regional vs Local LSTM Models for Short-term Streamflow Forecasting under Operational Constraints},
  author       = {Jorge Saavedra-Garrido, Jorge Arévalo, Luis De La Fuente, Aldo Tapia, Christopher Paredes-Arroyo, Ana María Córdova, Daira Velandia, Pablo Álvarez, Héctor Reyes-Serrano, Rodrigo Salas},
  year         = {2025},
  note         = {Manuscript under review},
  organization = {Proyecto FONDEF HidroCL}
}
```

---

**APA Citation (2025):**  
Saavedra-Garrido, J., Arévalo, J., De La Fuente, L., Tapia, A., Paredes-Arroyo, C., Córdova, A. M., Velandia, D., Álvarez, P., Reyes-Serrano, H., & Salas, R. (2025). *Regional vs Local LSTM Models for Short-term Streamflow Forecasting under Operational Constraints* [Manuscript under review]. Proyecto FONDEF HidroCL.

---

_Thank you for using this repository! Contributions and issues are welcome._  
