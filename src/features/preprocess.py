# src/features/preprocess.py
# Temporary path fix - remove later when we have proper package setup
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

    
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing for Telco Churn dataset.
    Returns: X_train, X_test, y_train, y_test, preprocessor (fitted ColumnTransformer)
    """
    df = df.copy()

    # 1. Drop useless column
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # 2. Handle TotalCharges: convert empty strings to NaN, then fill with 0 (new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)  # or df["MonthlyCharges"] * df["tenure"] if you want more accuracy

    # 3. Map target to binary (0/1)
    df[target_col] = df[target_col].map({"No": 0, "Yes": 1})

    # 4. Define feature types
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # Map Yes/No to 1/0 for binary features
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0})

    categorical_cols = [
        "gender", "MultipleLines", "InternetService", "Contract", "PaymentMethod"
    ]
    numerical_cols = [
        "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"
    ]

    # 5. Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            # Binary already mapped to 0/1, no further transform needed
        ],
        remainder="passthrough"  # keep any unmapped binary cols
    )

    # 6. Features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 7. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 8. Fit preprocessor on train only (leakage prevention)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print(f"Train shape after transform: {X_train_transformed.shape}")
    print(f"Test shape after transform: {X_test_transformed.shape}")

    return X_train, X_test, y_train, y_test, preprocessor, X_train_transformed, X_test_transformed


if __name__ == "__main__":
    from src.data.load_data import load_raw_data
    df = load_raw_data()
    X_train, X_test, y_train, y_test, preprocessor, Xt_train, Xt_test = preprocess_data(df)