import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values using linear interpolation and forward fill."""
    return df.interpolate(method='linear').ffill().bfill()

def winsorize_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Caps outliers using the 1.5 * IQR threshold."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    return df_clean

def normalize_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Applies Min-Max scaling to specified numerical columns."""
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled

if __name__ == "__main__":
    print("Agricultural data preprocessing pipeline loaded successfully.")
  
