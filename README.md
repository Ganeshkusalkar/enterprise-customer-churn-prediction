# Enterprise Telco Customer Churn Prediction Pipeline

Production-grade end-to-end MLOps project for predicting customer churn in telecom using the IBM Telco dataset.

[![CI](https://github.com/Ganeshkusalkar/enterprise-customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Ganeshkusalkar/enterprise-customer-churn-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![MLflow](https://img.shields.io/badge/MLflow-tracked-green)

## Problem Statement

Customer churn costs telecom companies millions annually in lost revenue and acquisition costs.  
This project builds a **reliable, production-ready system** to predict which customers are likely to churn — enabling targeted retention campaigns.

**Business Impact Goal**: Reduce churn rate by identifying at-risk customers early (simulated ~20–25% reduction potential).

## Solution Overview

End-to-end MLOps pipeline:
- Data validation & versioning
- Advanced feature engineering & preprocessing
- Model training with explainability & tracking
- Survival analysis + causal inference for deeper insights
- Real-time inference API + interactive dashboard
- Containerized & CI/CD automated

### Architecture Diagram

![MLOps Pipeline Architecture](docs/architecture-mlops.png)  
*End-to-end MLOps architecture: Data → Preprocessing → Training → Deployment → Monitoring → Retrain loop*

![Churn Prediction Flow](docs/churn-flow-diagram.png)  
*High-level churn prediction system flow (data ingestion to business action)*

![XGBoost Flowchart](docs/xgboost-flowchart.png)  
*How XGBoost builds boosted trees for accurate predictions*

## Key Features & Results

- **Data Versioning**: DVC for raw & processed data reproducibility
- **Validation**: Great Expectations programmatic suite
- **Preprocessing**: Modular scikit-learn ColumnTransformer (handles TotalCharges, categorical encoding, scaling)
- **Models**:
  - Baseline: Logistic Regression
  - Advanced: XGBoost (best performer)
  - Survival: Kaplan-Meier (Lifelines) for time-to-churn
  - Causal: DoWhy for root-cause analysis (e.g., monthly contract impact)
- **Tracking**: MLflow (experiments, metrics, artifacts)
- **Serving**: FastAPI real-time API + Streamlit interactive dashboard
- **Deployment**: Docker + GitHub Actions CI (lint & build)
- **Results** (from XGBoost run):
  - F1-Score: [Insert your value, e.g. 0.72]
  - PR-AUC: [Insert your value, e.g. 0.68]
  - Simulated retention value: ₹[Insert calculated value] Cr/year (assuming 20% intervention on predicted churners @ ₹12k/customer)

**Confusion Matrix** (XGBoost predictions):  
![Confusion Matrix](docs/confusion-matrix.png)

**Survival Curve** (Time-to-Churn):  
![Survival Curve](docs/survival-curve.png)

**MLflow Dashboard** (Experiments & Metrics):  
![MLflow Dashboard](docs/mlflow-dashboard.png)

## Tech Stack

- **Languages**: Python 3.10+
- **Data/ML**: pandas, scikit-learn, XGBoost, LightGBM, Lifelines, DoWhy, MLflow
- **Validation/Versioning**: Great Expectations, DVC
- **API**: FastAPI + Uvicorn
- **Dashboard**: Streamlit
- **Container/CI**: Docker, GitHub Actions
- **Other**: joblib, matplotlib, seaborn

## Setup & Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Ganeshkusalkar/enterprise-customer-churn-prediction.git
   cd enterprise-customer-churn-prediction