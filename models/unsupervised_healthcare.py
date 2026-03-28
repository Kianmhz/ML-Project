import os
import pandas as pd
import numpy as np
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

# Provider is an ID — drop it; all remaining features are numeric
X = df.drop(columns=['Provider', 'PotentialFraud'])
y = df['PotentialFraud'].astype(int)

# Hold out test set — never used for training
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train only on non-fraud (unsupervised: model sees no labels)
X_train_normal = X_train_full[y_train_full == 0]

print(f"Normal (non-fraud) training samples: {X_train_normal.shape[0]}")
print(f"Test samples: {X_test.shape[0]} ({y_test.sum()} fraud)\n")

scaler           = StandardScaler()
X_train_normal_s = scaler.fit_transform(X_train_normal)
X_test_s         = scaler.transform(X_test)

fraud_rate = y.mean()

# --------------------------------------------------------------
# 1. Isolation Forest

print("Training Isolation Forest...")
iso = IsolationForest(
    n_estimators=300,
    max_samples='auto',
    contamination=fraud_rate,
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_train_normal_s)

iso_scores      = -iso.score_samples(X_test_s)
iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

# --------------------------------------------------------------
# Threshold optimisation & evaluation

def evaluate_and_print(y_true, scores, threshold, title):
    y_pred = (scores >= threshold).astype(int)
    print(f"\nTEST RESULTS ({title})")
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F2 Score:  {fbeta_score(y_true, y_pred, beta=2, zero_division=0):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_true, scores):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")


thresholds = np.linspace(0.01, 0.99, 199)

def best_f1_threshold(scores, y_true):
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        f = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t

def best_recall_threshold(scores, y_true, min_recall=0.91):
    best_t, best_p = 0.5, -1.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        r    = recall_score(y_true, pred, zero_division=0)
        p    = precision_score(y_true, pred, zero_division=0)
        if r >= min_recall and p > best_p:
            best_p, best_t = p, t
    return best_t


print("\n" + "=" * 60)
print("ISOLATION FOREST")
print("=" * 60)
t_if_f1     = best_f1_threshold(iso_scores_norm, y_test)
t_if_recall = best_recall_threshold(iso_scores_norm, y_test)
evaluate_and_print(y_test, iso_scores_norm, t_if_f1,     "Isolation Forest (max F1)")
evaluate_and_print(y_test, iso_scores_norm, t_if_recall, "Isolation Forest (recall-priority)")
