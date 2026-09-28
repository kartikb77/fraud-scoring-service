import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

np.random.seed(42)

# Generate synthetic transaction data with realistic fraud patterns
n_samples = 5000

data = pd.DataFrame({
    'amount': np.random.exponential(scale=200, size=n_samples),
    'hour_of_day': np.random.randint(0, 24, size=n_samples),
    'transaction_velocity': np.random.poisson(lam=2, size=n_samples),  # transactions in last hour
    'is_new_device': np.random.binomial(1, 0.1, size=n_samples),
    'account_age_days': np.random.randint(1, 2000, size=n_samples),
})

# Fraud tends to correlate with: high amount, odd hours, high velocity, new device, young account
fraud_score = (
    (data['amount'] > 500).astype(int) * 0.3 +
    (data['hour_of_day'].between(0, 5)).astype(int) * 0.2 +
    (data['transaction_velocity'] > 4).astype(int) * 0.3 +
    data['is_new_device'] * 0.3 +
    (data['account_age_days'] < 30).astype(int) * 0.2
)

data['is_fraud'] = (fraud_score + np.random.normal(0, 0.15, n_samples) > 0.5).astype(int)

print(f"Fraud rate: {data['is_fraud'].mean():.2%}")

X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs('app/saved_models', exist_ok=True)
joblib.dump(model, 'app/saved_models/fraud_model.pkl')
print("Model saved to app/saved_models/fraud_model.pkl")