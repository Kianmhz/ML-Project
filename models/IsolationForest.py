import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, fbeta_score, roc_auc_score, confusion_matrix
)

# --------------------------------------------------------------
# Load data

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(base_dir, 'dataset', 'healthcare_fraud_preprocessed.csv'))

X = df.drop(columns=['Provider', 'PotentialFraud'])
y = df['PotentialFraud'].astype(int)

# Simple train/test split — no val set needed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Train samples: {len(X_train)}")
print(f"Test samples:  {len(X_test)} ({y_test.sum()} fraud)\n")

# --------------------------------------------------------------
# Scale — fit on train only

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --------------------------------------------------------------
# Train — fixed params, contamination='auto' (no label knowledge)

iso = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    max_features=1.0,
    contamination=0.35,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train_s)

# predict() returns -1 (anomaly) or 1 (normal) — convert to 0/1 fraud labels
y_pred      = (iso.predict(X_test_s) == -1).astype(int)
test_scores = -iso.score_samples(X_test_s)

# --------------------------------------------------------------
# Evaluate on TEST set

print("=" * 60)
print("ISOLATION FOREST (Unsupervised Baseline)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F2 Score:  {fbeta_score(y_test, y_pred, beta=2, zero_division=0):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, test_scores):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

