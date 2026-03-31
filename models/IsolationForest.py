import os
import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
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

# 3-way split: train (CV + final fit) / val (threshold tuning) / test (final eval)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

fraud_rate = y.mean()

print(f"Train samples:      {len(y_train)} ({(y_train==0).sum()} normal, {y_train.sum()} fraud)")
print(f"Validation samples: {len(y_val)} ({y_val.sum()} fraud)")
print(f"Test samples:       {len(y_test)} ({y_test.sum()} fraud)\n")

# --------------------------------------------------------------
# Hyperparameter grid

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_samples':  ['auto', 0.8],
    'max_features': [0.8, 1.0],
}

keys   = list(param_grid.keys())
combos = list(product(*param_grid.values()))

print(f"Tuning {len(combos)} hyperparameter combinations with 5-fold CV (metric: ROC AUC)...\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_params   = None
best_mean_auc = -1.0

for combo in combos:
    params = dict(zip(keys, combo))
    fold_aucs = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_tr = X_train.iloc[train_idx]
        y_fold_tr = y_train.iloc[train_idx]
        X_fold_vl = X_train.iloc[val_idx]
        y_fold_vl = y_train.iloc[val_idx]

        # Scale inside fold — fit on non-fraud only
        sc = StandardScaler()
        X_normal_s  = sc.fit_transform(X_fold_tr[y_fold_tr == 0])
        X_fold_vl_s = sc.transform(X_fold_vl)

        iso = IsolationForest(
            **params, contamination=fraud_rate, random_state=42, n_jobs=-1
        )
        iso.fit(X_normal_s)

        scores    = -iso.score_samples(X_fold_vl_s)
        fold_aucs.append(roc_auc_score(y_fold_vl, scores))

    mean_auc = np.mean(fold_aucs)
    folds_str = " | ".join(f"{a:.4f}" for a in fold_aucs)
    print(f"Params: {params} | Mean AUC: {mean_auc:.4f} | Folds: [{folds_str}]")

    if mean_auc > best_mean_auc:
        best_mean_auc = mean_auc
        best_params   = params

print(f"\nBest Params:      {best_params}")
print(f"Best CV Mean AUC: {best_mean_auc:.4f}\n")

# --------------------------------------------------------------
# Final model — fit on all non-fraud training samples

print("Training final Isolation Forest with best params...")
scaler       = StandardScaler()
X_normal_s   = scaler.fit_transform(X_train[y_train == 0])
X_val_s      = scaler.transform(X_val)
X_test_s     = scaler.transform(X_test)

iso = IsolationForest(
    **best_params, contamination=fraud_rate, random_state=42, n_jobs=-1
)
iso.fit(X_normal_s)

val_scores  = -iso.score_samples(X_val_s)
test_scores = -iso.score_samples(X_test_s)

# Normalise scores to [0, 1] using val+test range so scale is consistent
all_scores = np.concatenate([val_scores, test_scores])
s_min, s_max = all_scores.min(), all_scores.max()
val_scores_n  = (val_scores  - s_min) / (s_max - s_min)
test_scores_n = (test_scores - s_min) / (s_max - s_min)

# --------------------------------------------------------------
# Threshold tuning on VALIDATION set

thresholds = np.linspace(0.01, 0.99, 199)

def best_f2_threshold(scores, y_true):
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        f = fbeta_score(y_true, (scores >= t).astype(int), beta=2, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t

t_f2     = best_f2_threshold(val_scores_n, y_val)

val_f2_at_t = fbeta_score(y_val, (val_scores_n >= t_f2).astype(int), beta=2, zero_division=0)

print(f"Best threshold from VALIDATION (max F2):         {t_f2:.4f}  (val F2 = {val_f2_at_t:.4f})")

# --------------------------------------------------------------
# Evaluate on TEST set

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

print("\n" + "=" * 60)
print("ISOLATION FOREST")
print("=" * 60)
evaluate_and_print(y_test, test_scores_n, t_f2,     "Isolation Forest (max F2)")
