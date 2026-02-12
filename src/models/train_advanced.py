# src/models/train_advanced.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_recall_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
from dowhy import CausalModel
from src.features.preprocess import preprocess_data  # reuse


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(save_path)
    plt.close()
    return save_path

def train_xgboost_and_log():
    df_raw = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test, preprocessor, Xt_train, Xt_test = preprocess_data(df_raw)

    mlflow.set_experiment("telco-churn-prediction")
    with mlflow.start_run(run_name="xgboost-advanced"):
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
            random_state=42,
            eval_metric="aucpr"
        )
        model.fit(Xt_train, y_train)

        y_pred = model.predict(Xt_test)
        y_prob = model.predict_proba(Xt_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)

        print(f"XGBoost F1: {f1:.4f}")
        print(f"XGBoost PR-AUC: {pr_auc:.4f}")
        print(classification_report(y_test, y_pred))

        # Log
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.xgboost.log_model(model, "xgboost_model")

        # Artifact: confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact(cm_path)

        # Simple churn reduction simulation (assume 20% of predicted churners retained)
        predicted_churners = sum(y_pred == 1)
        retained = predicted_churners * 0.20
        value_saved = retained * 12000  # assume ₹12k per customer
        mlflow.log_metric("simulated_value_saved_rs", value_saved)

def survival_analysis():
    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    kmf = KaplanMeierFitter()
    kmf.fit(df["tenure"], event_observed=df["Churn"])
    
    plt.figure(figsize=(8,6))
    kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curve (Time to Churn)")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Survival Probability")
    survival_plot = "survival_curve.png"
    plt.savefig(survival_plot)
    plt.close()

    # Log to MLflow (in a separate run or same)
    with mlflow.start_run(run_name="survival_analysis", nested=True):
        mlflow.log_artifact(survival_plot)
        mlflow.log_text("Kaplan-Meier fitted for time-to-churn", "survival_summary.txt")

def causal_inference_example():
    # Simplified DoWhy example: does "Contract" cause churn? (Monthly vs Two year)
    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df[["Contract", "MonthlyCharges", "tenure", "Churn"]].copy()
    df["treatment"] = (df["Contract"] == "Month-to-month").astype(int)  # treatment = monthly contract

    model = CausalModel(
        data=df,
        treatment="treatment",
        outcome="Churn",
        common_causes=["MonthlyCharges", "tenure"]
    )

    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_matching"
    )

    print(estimate)

    with mlflow.start_run(run_name="causal_inference", nested=True):
        mlflow.log_text(str(estimate), "causal_estimate.txt")
        mlflow.log_text("Causal effect of monthly contract on churn", "causal_summary.txt")

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    train_xgboost_and_log()
    survival_analysis()
    causal_inference_example()