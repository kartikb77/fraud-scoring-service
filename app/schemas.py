from pydantic import BaseModel

class TransactionFeatures(BaseModel):
    amount: float
    hour_of_day: int
    transaction_velocity: int
    is_new_device: int
    account_age_days: int

class FraudScoreResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    decision: str  # "APPROVE", "REVIEW", or "BLOCK"