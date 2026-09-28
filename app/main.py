from fastapi import FastAPI
from app.schemas import TransactionFeatures, FraudScoreResponse
from app.model import predict_fraud

app = FastAPI(title="Fraud Scoring Service")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score-transaction", response_model=FraudScoreResponse)
def score_transaction(features: TransactionFeatures):
    result = predict_fraud(features.model_dump())
    return result