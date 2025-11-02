from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="Fraud Detection API",
    description="Predict fraud probability for a given bank transaction.",
    version="1.0.0"
)

model = joblib.load("SVM\model.pkl")
scaler = joblib.load("SVM\scaler.pkl")
# scaler = None

# Define input schema
class Transaction(BaseModel):
    TransactionAmount: float
    TransactionDuration: float
    LoginAttempts: int
    AccountBalance: float
    CustomerAge: int

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running."}

@app.post("/SVM/predict")
def score(transaction: Transaction):
    data = transaction.model_dump()

    # Extract and scale features
    features = np.array([
        data["TransactionAmount"],
        data["TransactionDuration"],
        data["LoginAttempts"],
        data["AccountBalance"],
        data["CustomerAge"]
    ]).reshape(1, -1)

    scaled_features = scaler.transform(features)

    # Get prediction
    pred = model.predict(scaled_features)[0]
    prob = getattr(model, "predict_proba", None)
    score = float(prob(scaled_features)[0][1]) if prob else float(pred)

    return {
        "fraud_prediction": int(pred),
        "fraud_score": round(score, 4)
    }

