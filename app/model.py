import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "saved_models" / "fraud_model.pkl"
_model = joblib.load(MODEL_PATH)

def predict_fraud(features: dict) -> dict:
    df = pd.DataFrame([features])
    probability = _model.predict_proba(df)[0][1]  # probability of class 1 (fraud)

    if probability >= 0.7:
        decision = "BLOCK"
    elif probability >= 0.4:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    return {
        "is_fraud": bool(probability >= 0.5),
        "fraud_probability": round(float(probability), 4),
        "decision": decision
    }