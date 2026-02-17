<div align="center">

# 🔮 Enterprise Telco Customer Churn Prediction Pipeline

### Production-grade MLOps · Real-time API · Interactive Dashboard · Causal Insights

[![CI - Lint & Docker Build](https://github.com/Ganeshkusalkar/enterprise-customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Ganeshkusalkar/enterprise-customer-churn-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-Versioned-945DD6?logo=dvc&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**[🚀 Live API (Swagger)](https://enterprise-customer-churn-prediction.onrender.com/docs)**  
**[📊 Live Interactive Dashboard](https://enterprise-customer-churn-prediction-teh5bckdms9u6dgbguu8hs.streamlit.app)**  
**[GitHub Repo](https://github.com/Ganeshkusalkar/enterprise-customer-churn-prediction)**

> *"Acquiring a new customer costs 5–7× more than retaining an existing one."*  
> This production system identifies at-risk customers early — so businesses can act before they leave.

</div>

---

## 📌 Quick Links

- [Executive Summary](#executive-summary)  
- [Business Impact](#business-impact)  
- [Live Demos](#live-demos)  
- [Architecture](#architecture)  
- [Model Performance](#model-performance)  
- [How to Run Locally](#how-to-run-locally)  
- [Tech Stack](#tech-stack)  
- [Author & Contact](#author--contact)

---

## Executive Summary

This is **not** another Jupyter notebook churn model.

This is a **full production-grade MLOps pipeline** for telecom customer churn prediction, including:

- Data versioning & quality gates  
- Advanced modeling (XGBoost + survival + causal inference)  
- Experiment tracking  
- Real-time REST API  
- Interactive business dashboard  
- Docker containerization  
- GitHub Actions CI/CD  
- Live deployment on Render & Streamlit Cloud

Built to be **reproducible**, **auditable**, and **business-useful** — exactly what hiring managers at fintech/telecom/product companies look for.

---

## Business Impact

**Dataset**: IBM Telco Customer Churn (7,043 customers, 26.5% churn rate)

**Simulated retention value** (conservative assumptions):

- Average customer LTV: ₹12,000 / year  
- Identified high-risk customers: ~450 (top 20% predicted churners)  
- Retention success rate after intervention: 40%  
- **Estimated annual saved revenue**: **₹2.16 Crore**

**Key insights delivered by the system**:

- Month-to-month contracts increase churn risk by **~3.4×** (causal effect via DoWhy)  
- Fiber optic + electronic check combination is the highest-risk segment  
- Highest churn hazard window: **months 3–12** of tenure (survival curves)

---

## Live Demos

> ⚡ First request may take 10–30 seconds (Render free tier cold start)

| Service                     | Link                                                                                 | What it does                                      |
|-----------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------|
| **API – Swagger UI**        | [https://enterprise-customer-churn-prediction.onrender.com/docs](https://enterprise-customer-churn-prediction.onrender.com/docs) | Real-time /predict endpoint – test predictions    |
| **Interactive Dashboard**   | [https://enterprise-customer-churn-prediction-teh5bckdms9u6dgbguu8hs.streamlit.app](https://enterprise-customer-churn-prediction-teh5bckdms9u6dgbguu8hs.streamlit.app) | Fill customer profile → instant churn risk score  |

---

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/Ganeshkusalkar/enterprise-customer-churn-prediction/main/docs/mlops-pipeline.png" width="820" alt="End-to-end MLOps Architecture">
  <br><em>Complete MLOps lifecycle: Data → Validation → Training → Serving → Monitoring</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Ganeshkusalkar/enterprise-customer-churn-prediction/main/docs/churn-flow.png" width="720" alt="Business Churn Flow">
  <br><em>From customer event → risk scoring → retention action</em>
</p>

---

## Model Performance

**Primary model**: XGBoost (production choice)

| Metric                | Value  | Notes                                      |
|-----------------------|--------|--------------------------------------------|
| **F1-Score**          | **0.72** | Primary metric (imbalanced class)          |
| **PR-AUC**            | **0.68** | Strong probability ranking                 |
| Accuracy              | 0.78   | Secondary (73% majority class)             |
| Recall (Churn class)  | ~0.74  | Catches most actual churners               |

**Confusion Matrix**  
<img src="https://github.com/Ganeshkusalkar/enterprise-customer-churn-prediction/blob/main/docs/confusion_matrix_xgboost.png" width="420" alt="Confusion Matrix">

**Survival Curve (by Contract Type)**  
<img src="https://raw.githubusercontent.com/Ganeshkusalkar/enterprise-customer-churn-prediction/main/docs/survival-curve.png" width="520" alt="Kaplan-Meier Survival Curve">

---

## How to Run Locally

### Easiest way (one command)

```bash
git clone https://github.com/Ganeshkusalkar/enterprise-customer-churn-prediction.git
cd enterprise-customer-churn-prediction

# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# or source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Pull versioned data + artifacts
dvc pull

# Launch API + Dashboard together
python run_all.py
