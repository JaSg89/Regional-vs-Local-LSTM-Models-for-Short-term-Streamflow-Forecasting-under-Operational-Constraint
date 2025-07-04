# 📂 Local Models (LOCAL MODELS)

This directory contains the scripts required to prepare data, generate sequences, and perform forecasts using the **local LSTM models**, each trained independently for a specific catchment (`gauge_id`). Unlike the regional approach, each model processes only the information from its own catchment.

---

## 🗄️ Folder Structure

```plaintext
LOCAL_MODELS/
├── LOCAL_SEQ.py             # Generates and saves training, validation, and test sequences
├── FORECAST.py             # Runs forecasts and evaluates metrics on test data
├── LSTM_LOCAL2.CSV         # Input file with raw data for all catchments
└── Trained_Models_TEST/    # Output from both scripts, organized by gauge_id
    ├── Sequences/         # HDF5 sequences and auxiliary arrays (.npy)
    │   └── {gauge_id}/
    │       ├── Sequences_train/train.h5
    │       ├── Sequences_val/val.h5
    │       ├── Sequences_test/test.h5
    │       ├── gauge_ids_train.npy
    │       ├── dates_train.npy
    │       └── ...
    ├── Scaler/             # Serialized scalers for each catchment
    │   └── {gauge_id}/
    │       ├── X_scaler_{gauge_id}.pkl
    │       └── y_scaler_{gauge_id}.pkl
    └── Models/             # Trained models and forecast results
        └── {gauge_id}/
            ├── best_model.h5
            ├── y_pred_inv_test.npy
            ├── y_test_inv.npy
            ├── Qmean_LSTM_p0d.csv
            ├── kge_test_p0d.csv
            └── ...
```

---

## ⚙️ Dependencies

- Python 3.8+  
- pandas==1.5.3  
- numpy==1.25.0  
- scipy==1.11.3  
- h5py==1.12.2  
- scikit-learn==1.2.2  
- tensorflow==2.11.0 (GPU optional)  
- keras==2.11.0  
- statsmodels==0.14.0  
- CRPS.CRPS==2.0.4  
- pickle (standard library)  
- logging (standard library)  

_Using a Conda environment or virtualenv is recommended to manage these dependencies._

---

## 🔧 Configuration

1. **Update paths** in both scripts (`LOCAL_SEQ.py` and `FORECAST.py`):  
   - `dir_path` should point to the root `Trained_Models_TEST/` folder.  
   - `LSTM_LOCAL2.CSV` must be located at the specified path.  
2. **Random seed** (`SEED`): both scripts fix the seed to ensure reproducibility.  
3. **GPU configuration**: in `FORECAST.py`, set `CUDA_VISIBLE_DEVICES` as needed.  
4. **Sequence parameters**:  
   - `time_steps` (default 250): number of days used as input.  
   - `prediction_days` (default 5): forecast horizon.  

---

## 🚀 Usage

### 1. Generate Sequences

```bash
python LOCAL_SEQ.py
```

- Reads `LSTM_LOCAL2.CSV`, filters out invalid catchments (`mask2 == 3`), performs feature engineering and normalization.  
- Splits data into training, validation, and test sets based on `caudal_mask2_p0d` (0, 1, 2).  
- Scales data with `MinMaxScaler` (fit only on training) and saves the scalers.  
- Creates HDF5 sequence files (`.h5`) and auxiliary arrays (`.npy`) for each catchment.  

### 2. Forecast and Evaluate

```bash
python FORECAST.py
```

- Loads the best model (`best_model.h5`) and scalers for each catchment.  
- Runs Monte Carlo Dropout (`num_simulations`) to estimate uncertainty.  
- Inverts scaling and power transformation (root⁻¹) to recover actual streamflow values.  
- Computes daily metrics: mean, standard deviation, median, kurtosis, skewness, percentiles, **KGE**, and **CRPS**.  
- Saves results in `.npy` and `.csv` files under `Models/{gauge_id}/`.  

---

## 📈 Output and Metrics

- **Sequences**: HDF5 files with `X_{train,val,test}` and `y_{train,val,test}`.  
- **Scalers**: pickle files `X_scaler_{gauge_id}.pkl`, `y_scaler_{gauge_id}.pkl`.  
- **Predictions**: NumPy arrays `y_pred_inv_test.npy`, `y_test_inv.npy`.  
- **Forecast CSVs**: `Qmean_LSTM_p{0..4}d.csv` with statistics and observations.  
- **Efficiency CSVs**: `kge_test_p{0..4}d.csv` with KGE per catchment.  

---

## 📝 Notes & Best Practices

- **Temporal validation**: `caudal_mask2_p0d` ensures no data leakage between splits.  
- **Quality control**: incomplete sequences are skipped; logs count of omitted and created sequences.  
- **Error handling**: `FORECAST.py` logs robust statistics for NaNs and outliers.  
- **Extensibility**: to change the forecast horizon or sequence length, adjust `prediction_days` and `time_steps`.  
- **Optimization**: consider using callbacks (`EarlyStopping`, `ReduceLROnPlateau`) during local training.  

---

_If you have suggestions or find bugs, please open an issue._  
