# Fraud Scoring Service

A lightweight machine learning microservice built with FastAPI and Scikit-Learn to assess transaction fraud risks in real-time.

---

## Features

- **Real-Time Fraud Evaluation**: Scikit-Learn `RandomForestClassifier` trained on transaction features (transaction amount, hour of day, transaction velocity, new device indicator, account age).
- **FastAPI Endpoints**: High-performance REST endpoints for health checks and transaction fraud scoring with automatic validation via Pydantic.
- **Decision Engine**: Automatically classifies transactions into `APPROVE`, `REVIEW`, or `BLOCK` actions based on predicted risk thresholds.

---

## API Endpoints

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "ok"
}
```

### 2. Score Transaction
- **Endpoint**: `POST /score-transaction`
- **Request Body**:
```json
{
  "amount": 250.50,
  "hour_of_day": 3,
  "transaction_velocity": 5,
  "is_new_device": 1,
  "account_age_days": 12
}
```
- **Response**:
```json
{
  "is_fraud": true,
  "fraud_probability": 0.8421,
  "decision": "BLOCK"
}
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- `pip`

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kartikb77/fraud-scoring-service.git
   cd fraud-scoring-service
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Train or retrain the model**:
   ```bash
   python train_model.py
   ```

5. **Run the API server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Interactive Documentation**:
   Visit `http://localhost:8000/docs` to test endpoints via Swagger UI.

---

## Configuration

Copy `.env.example` to `.env` to configure environment-specific variables:
```bash
cp .env.example .env
```
Ensure real secrets and sensitive keys are never committed to version control.
