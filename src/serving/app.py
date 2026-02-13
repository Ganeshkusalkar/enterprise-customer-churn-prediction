# src/serving/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Production-grade XGBoost-based churn prediction service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Paths
PREPROCESSOR_PATH = Path("data/processed/preprocessor.pkl")
MODEL_PATH = Path("models/xgboost_model.pkl")

# Global artifacts
preprocessor = None
model = None

@app.on_event("startup")
async def startup_event():
    global preprocessor, model
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
        logger.info("Model and preprocessor loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

@app.get("/health")
async def health_check():
    if preprocessor is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "API live", "docs": "/docs", "predict": "/predict"}

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict", response_model=Dict[str, Any])
async def predict(customer: Customer):
    if preprocessor is None or model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([customer.dict()])

        # === IMPORTANT: Apply same binary mapping as in training ===
        binary_cols = [
            "Partner", "Dependents", "PhoneService", "PaperlessBilling",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]

        for col in binary_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].map({
                    "Yes": 1,
                    "No": 0,
                    "No internet service": 0,
                    "No phone service": 0
                }).fillna(0).astype(int)

        # Now preprocess (scaling + one-hot)
        transformed = preprocessor.transform(input_df)

        # Predict
        prob = model.predict_proba(transformed)[0][1]
        prediction = "Yes" if prob >= 0.5 else "No"

        return {
            "churn_probability": round(float(prob), 4),
            "predicted_churn": prediction,
            "confidence": round(float(max(model.predict_proba(transformed)[0])), 4),
            "note": "Probability ≥ 0.5 → predicted to churn"
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)