# 📂 Data Imputation

The `Imputation/` directory groups together the tools needed for **operational imputation** of hydrological, climatic, and geographic data.

## 1. Directory Structure

```plaintext
Imputation/
├── databases/              # Raw, unprocessed data (time series)
├── DFs_IMPUTED/            # Historical imputed variables (2000–31 Dec 2023)
├── DFs_UPDATED/            # Operational results (daily imputations)
├── IMPUTATION_CLASS.py     # Core logic: HidroCL_Complete class
└── Imputation_2000_2023.py # Initial generation of base files
```

- **databases/**: Input folder containing all raw time series data.
- **DFs_IMPUTED/**: Static imputed variables covering the period 2000–2023.
- **DFs_UPDATED/**: Output of operational imputations; updated daily.
- **IMPUTATION_CLASS.py**: Defines the `HidroCL_Complete` class, responsible for detecting and filling missing values.
- **Imputation_2000_2023.py**: Static script to generate the initial files in `DFs_IMPUTED/`.

## 2. Dependencies

- Python 3.x
- pandas==2.1.1
- numpy==1.26.0
- scipy==1.11.3
- h5py
- python-dotenv
- Standard modules: `os`, `math`, `pickle`, `json`, `time`, `shutil`, `warnings`

Quick install:
```bash
pip install pandas==2.1.1 numpy==1.26.0 scipy==1.11.3 h5py python-dotenv
```

## 3. Environment Setup

Create a `.env` file in the project root with the following variables:
```
IMPUTED_PATH=/path/to/DFs_IMPUTED
UPDATED_PATH=/path/to/DFs_UPDATED
```

## 4. Running Imputation

```python
from IMPUTATION_CLASS import HidroCL_Complete
from dotenv import load_dotenv
import os

# Load paths from .env
load_dotenv()
path_read = os.getenv('IMPUTED_PATH')
path_save = os.getenv('UPDATED_PATH')

# Lists of variable codes to process
list_observed = [
    'pp_o_era5_pp_mean_b_none_d1_p0d',
    'tmp_o_era5_tmp_mean_b_none_d1_p0d',
    # ... more observed variables
]
list_forecasted = [
    'tmp_f_gfs_tmp_mean_b_none_d1_p0d',
    'tmp_f_gfs_tmp_max_b_none_d1_p0d',
    # ... more forecasted variables
]

# Instantiate and run
variable = HidroCL_Complete('tmp_f_gfs_tmp_min_b_none_d1_p0d')
variable.get_imputation(varcodes_list=list_observed, path_save_base=path_save, path_read=path_read)
variable.get_imputation(varcodes_list=list_forecasted, path_save_base=path_save, path_read=path_read)
```

### Input Parameters

- **varcodes_list** (`List[str]`): List of variable codes to impute.
- **path_read** (`str`): Path to `DFs_IMPUTED/`.
- **path_save_base** (`str`): Path to `DFs_UPDATED/`.

### Output

CSV files with updated time series up to the execution date, organized into `observed/` and `forecasted/` subfolders within `DFs_UPDATED/`.

## 5. License

This project is licensed under the **Apache 2.0** license.
