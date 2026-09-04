# agricultural-data-preprocessing-ml
# Agricultural Data Preprocessing Pipeline

A standardized Python data-cleaning pipeline designed to aggregate, clean, and preprocess multi-source agricultural datasets (such as **FAOSTAT**, **NASA POWER**, and **Agmarknet**) for machine learning models.

---

## Features

- **Missing Data Imputation**: Fills temporal gaps in environmental and crop data using linear interpolation and forward/backward fills.
- **Outlier Capping**: Detects and caps extreme sensor glitches or anomalous data points using the Interquartile Range (IQR) method.
- **Feature Normalization**: Scales numerical variables (e.g., rainfall, temperature, crop yield) to a uniform range using `MinMaxScaler`.

---

## Repository Structure

```text
├── .gitignore           # Excludes virtual environments & temporary cache files
├── README.md            # Project documentation
├── preprocess.py        # Core python data preprocessing functions
└── requirements.txt     # Python dependencies

## clone the repository:
'''bash
git clone [https://github.com/pratzyio/agricultural-data-preprocessing-ml.git]
cd agricultural-data-preprocessing-ml
'''

## Install required Packages:
'''bash
pip install -r requirements.txt
'''

## Usage:
'''python
import pandas as pd
from preprocess import handle_missing_data, winsorize_iqr, normalize_features

# 1. Load your raw dataset
df = pd.read_csv("your_agricultural_data.csv")

# 2. Clean missing values
df_clean = handle_missing_data(df)

# 3. Cap outliers for numerical features
features = ["temperature", "rainfall", "crop_yield"]
df_capped = winsorize_iqr(df_clean, columns=features)

# 4. Normalize numerical columns
df_final = normalize_features(df_capped, columns=features)
'''
