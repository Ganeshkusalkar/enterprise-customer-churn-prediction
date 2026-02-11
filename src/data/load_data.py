# src/data/load_data.py
import pandas as pd
from pathlib import Path

def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.
    This function is meant to be used consistently across notebooks and pipelines.
    """
    data_path = Path("data/raw/Telco-Customer-Churn.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run DVC pull if needed.")
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head(3))
    print("\nColumns:", df.columns.tolist())