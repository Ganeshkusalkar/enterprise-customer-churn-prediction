# src/serving/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from typing import List

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Production-grade churn prediction service with XGBoost",
    version="1.0.0"
)

# Load preprocessor and model (from processed artifacts)
MODEL_PATH = Path("data/processed/preprocessor.pkl")
PREPROCESSOR_PATH = Path("data/processed/preprocessor.pkl")  # same fitted one
XGB_MODEL_PATH = "models/xgboost_model"  # we'll save it properly later

preprocessor = joblib.load(PREPROCESSOR_PATH)
# For now use logistic as fallback; replace with XGBoost after saving
model = joblib.load("models/xgboost_churn_model.pkl")  # ← change to XGBoost later

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int                # this is 0/1 → int OK
    Partner: str                      # "Yes"/"No"
    Dependents: str                   # "Yes"/"No"
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

@app.get("/")
def root():
    return {"message": "Telco Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Apply the SAME binary mappings as in training (critical!)
        binary_cols = [
            "Partner", "Dependents", "PhoneService", "PaperlessBilling",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]

        mapping = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
        
        for col in binary_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].map(mapping).fillna(0).astype(int)

        # Now safe to transform
        input_transformed = preprocessor.transform(input_df)

        # Predict
        prob = model.predict_proba(input_transformed)[0][1]
        prediction = "Yes" if prob > 0.5 else "No"

        return {
            "churn_probability": round(float(prob), 4),
            "predicted_churn": prediction,
            "confidence": round(float(max(model.predict_proba(input_transformed)[0])), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)