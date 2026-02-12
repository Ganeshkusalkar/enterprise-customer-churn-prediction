# src/models/train_baseline.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc, confusion_matrix, classification_report
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.features.preprocess import preprocess_data  # reuse your preprocessor

def train_and_log_baseline():
    # Load raw data and preprocess (reuses your function)
    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")  # or use load_raw_data()
    X_train, X_test, y_train, y_test, preprocessor, Xt_train, Xt_test = preprocess_data(df)

    # Start MLflow run
    mlflow.set_experiment("telco-churn-prediction")
    with mlflow.start_run(run_name="baseline-logistic"):
        # Model
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        model.fit(Xt_train, y_train)

        # Predict
        y_pred = model.predict(Xt_test)
        y_prob = model.predict_proba(Xt_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Log to MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.sklearn.log_model(model, "logistic_model")
        mlflow.log_artifact("notebooks/01_eda.ipynb")  # example artifact

        # Save locally too
        joblib.dump(model, "models/logistic_baseline.pkl")
        print("Baseline model trained and logged!")

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    train_and_log_baseline()