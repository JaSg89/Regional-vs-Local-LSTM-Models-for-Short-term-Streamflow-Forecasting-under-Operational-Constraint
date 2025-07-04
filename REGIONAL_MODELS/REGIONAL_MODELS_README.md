# 📂 Regional Models (REGIONAL MODELS)

This directory contains the scripts and artifacts for training, forecasting, and evaluating the **regional LSTM model**, which is trained once on the combined data of all catchments (`gauge_id`). This approach leverages shared hydrological patterns across sites and provides a global forecast.

---

## 🗄️ Directory Structure

```plaintext
REGIONAL_MODELS/
├── SEQUENCES.py           # Prepares and scales global data; creates train/val/test sequences
├── FORECAST.py            # Trains or loads the regional model; runs forecasts and computes metrics
├── LSTM11.csv             # Raw input data for all catchments
└── Trained_Models_TEST/   # Centralized output directory with sequences, scalers, model, and results
    ├── Secuencias_train/
    │   └── train.h5            # HDF5 dataset for training (X_train, y_train)
    ├── Secuencias_val/
    │   └── val.h5              # HDF5 dataset for validation (X_val, y_val)
    ├── Secuencias_test/
    │   └── test.h5             # HDF5 dataset for testing (X_test, y_test)
    ├── X_scaler.pkl           # Global feature scaler (MinMaxScaler)
    ├── y_scaler.pkl           # Global target scaler (MinMaxScaler)
    ├── num_sequences.npy      # Array with [train_count, val_count, test_count]
    ├── best_model.h5          # Trained regional LSTM model
    ├── Qmean_LSTM_p0d.csv     # Forecast and metrics for day 0
    ├── Qmean_LSTM_p1d.csv     # Forecast and metrics for day 1
    └── ...                    # Additional CSVs for days 2–4 and summary metrics
```

---

## ⚙️ Dependencies

- Python 3.7+
- pandas==1.5.3
- numpy==1.25.0
- scipy==1.11.3
- statsmodels==0.14.0
- h5py==1.12.2
- scikit-learn==1.2.2
- tensorflow==2.11.0
- keras==2.11.0
- CRPS.CRPS==2.0.4
- pickle (standard library)
- logging (standard library)

*Use a virtual environment or Conda to isolate these dependencies.*

---

## 🔧 Configuration

1. **Update data paths** at the top of both scripts (`SEQUENCES.py`, `FORECAST.py`):
   - Set `dir_path` to point to `REGIONAL_MODELS/Trained_Models_TEST/`.
   - Ensure the raw CSV (`LSTM11.csv`) is referenced correctly.
2. **Random seeds**: Both scripts fix `random.seed(123)`, `np.random.seed(123)`, and `tf.random.set_seed(123)` for reproducibility.
3. **GPU selection**: In `FORECAST.py`, configure `CUDA_VISIBLE_DEVICES` as needed.
4. **Sequence settings** (in `SEQUENCES.py`):
   - `time_steps` (default 250 days) — input sequence length.
   - `prediction_days` (default 5 days) — forecast horizon.

---

## 🚀 Workflow

### 1. Generate Global Sequences

```bash
python SEQUENCES.py
```

- **Load** the full `LSTM11.csv` dataset and convert `date` column to datetime.
- **Feature engineering**: compute SWE (`snw_o_era5_swe`), aggregate glacial variables, and transform target `caudal_mean_p0d` with a fourth-root.
- **Temporal filtering**: remove max/mean and mask columns for p1–p4; split by `caudal_mask2_p0d` (0=train | 1=val | 2=test).
- **Global scaling**: fit `MinMaxScaler` on the **entire** training split; apply to val and test.
- **Save scalers**: `X_scaler.pkl`, `y_scaler.pkl` in `Trained_Models_TEST/`.
- **Count sequences**: determine valid start–end windows for each `gauge_id`.
- **Create HDF5**: write `train.h5`, `val.h5`, and `test.h5` with X and y arrays; save `gauge_ids` and `dates` per split.
- **Record counts**: store `[n_train, n_val, n_test]` in `num_sequences.npy`.

### 2. Train & Forecast

```bash
python FORECAST.py
```

- **Load** `train.h5`, `val.h5`, `test.h5` via a custom `DataGenerator` (inherits from Keras `Sequence`).
- **Build or load** the LSTM model (`best_model.h5` if already trained).
- **Monte Carlo Dropout**: run `num_simulations` forward passes on test split to quantify uncertainty.
- **Inverse transform**: apply `y_scaler.inverse_transform` and raise to the fourth power.
- **Compute metrics** per day and per `gauge_id`:
  - Ensemble stats: mean, std, median, kurtosis, skewness, percentiles (5, 25, 75, 95).
  - **KGE** (Kling–Gupta Efficiency) and **CRPS** (Continuous Ranked Probability Score).
- **Save outputs**:
  - Numpy arrays: `y_test_inv.npy`, `y_pred_inv_test.npy`.
  - CSV files: `Qmean_LSTM_p{0..4}d.csv` for forecasts + metrics.
  - Aggregate KGE summaries printed to console.

---

## 📈 Output Files

- **HDF5 sequences**: X and y for train/val/test splits.
- **Scalers**: `X_scaler.pkl`, `y_scaler.pkl`.
- **Model**: `best_model.h5`.
- **Sequence counts**: `num_sequences.npy`.
- **Forecast arrays**: `y_test_inv.npy`, `y_pred_inv_test.npy`.
- **Forecast CSVs**: `Qmean_LSTM_p{0..4}d.csv`.
- **Efficiency CSVs**: integrated in the same files via columns `KGE` and `crps`.

---

## 📝 Notes & Best Practices

- **Global vs. local scaling**: ensure a single scaler fits all catchments to maintain comparability.
- **Temporal splits**: use `caudal_mask2_p0d` to prevent data leakage across train/val/test.
- **DataGenerator**: closes HDF5 files in `__del__`; safe for large datasets.
- **Vectorized metrics**: compute KGE and CRPS in bulk for efficiency.
- **Callbacks**: integrate `EarlyStopping` and `ReduceLROnPlateau` during training to avoid overfitting.

---

*For issues or contributions, please open an issue on the repository.*
