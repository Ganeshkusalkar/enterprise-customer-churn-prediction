# src/data/save_splits.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
from pathlib import Path
from src.features.preprocess import preprocess_data
from src.data.load_data import load_raw_data

if __name__ == "__main__":
    df = load_raw_data()
    X_train, X_test, y_train, y_test, preprocessor, Xt_train, Xt_test = preprocess_data(df)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)

    joblib.dump(X_train, processed_dir / "X_train.pkl")
    joblib.dump(X_test, processed_dir / "X_test.pkl")
    joblib.dump(y_train, processed_dir / "y_train.pkl")
    joblib.dump(y_test, processed_dir / "y_test.pkl")
    joblib.dump(preprocessor, processed_dir / "preprocessor.pkl")

    print("Processed splits and preprocessor saved!")