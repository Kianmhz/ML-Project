import os
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, fbeta_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

# --------------------------------------------------------------
# Load data

base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, 'dataset', 'healthcare_fraud_preprocessed.csv')
df           = pd.read_csv(dataset_path)

# Provider is an ID — drop it, keep everything else as features
X = df.drop(columns=['Provider', 'PotentialFraud'])
y = df['PotentialFraud'].astype(int)

# Hold out test set — never touched during tuning
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train+Val set: {X_train_full.shape[0]:,} samples ({y_train_full.sum()} fraud)")
print(f"Test set:      {X_test.shape[0]:,} samples ({y_test.sum()} fraud)\n")

# --------------------------------------------------------------
# Hyperparameter grid

param_grid = {
    'max_depth':        [4, 6, 8],
    'learning_rate':    [0.05, 0.1],
    'subsample':        [0.75, 0.9],
    'min_child_weight': [3, 5, 10],
    'reg_lambda':       [1.0, 3.0],
}

keys   = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))

print(f"Tuning {len(combos)} hyperparameter combinations with 5-fold CV (metric: F1)...\n")

kf           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
thresholds   = np.linspace(0.05, 0.95, 50)
best_mean_f1 = -1.0
best_params  = None

for combo in combos:
    params   = dict(zip(keys, combo))
    fold_f1s = []

    for tr_idx, vl_idx in kf.split(X_train_full, y_train_full):
        X_tr = X_train_full.iloc[tr_idx]
        y_tr = y_train_full.iloc[tr_idx]
        X_vl = X_train_full.iloc[vl_idx]
        y_vl = y_train_full.iloc[vl_idx]

        dist             = y_tr.value_counts().sort_index()
        scale_pos_weight = dist[0] / dist[1]

        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_estimators=500,
            early_stopping_rounds=30,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            tree_method='hist',
            verbosity=0,
            **params,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

        proba   = model.predict_proba(X_vl)[:, 1]
        fold_f1 = max(
            f1_score(y_vl, (proba >= t).astype(int), zero_division=0)
            for t in thresholds
        )
        fold_f1s.append(fold_f1)

    mean_f1 = np.mean(fold_f1s)
    print(f"Params: {params} | Mean F1: {mean_f1:.4f} | Folds: {[f'{f:.4f}' for f in fold_f1s]}")

    if mean_f1 > best_mean_f1:
        best_mean_f1 = mean_f1
        best_params  = params

print(f"\nBest Params:     {best_params}")
print(f"Best CV Mean F1: {best_mean_f1:.4f}\n")

# --------------------------------------------------------------
# Final model — retrain on full training data with best params

X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
)

dist             = y_tr2.value_counts().sort_index()
scale_pos_weight = dist[0] / dist[1]

final_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=1000,
    early_stopping_rounds=50,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    verbosity=0,
    **best_params,
)
final_model.fit(X_tr2, y_tr2, eval_set=[(X_val2, y_val2)], verbose=False)

print(f"Best iteration:   {final_model.best_iteration}")
print(f"Best val ROC-AUC: {float(final_model.best_score):.6f}")

# --------------------------------------------------------------
# Threshold optimisation on val set

y_val_proba       = final_model.predict_proba(X_val2)[:, 1]
min_recall_target = 0.91

# A) Max-F1
best_threshold = 0.5
best_val_f1    = -1.0
for t in thresholds:
    f = f1_score(y_val2, (y_val_proba >= t).astype(int), zero_division=0)
    if f > best_val_f1:
        best_val_f1, best_threshold = f, t

print(f"\nBest threshold (max F1):  {best_threshold:.4f}  |  Val F1: {best_val_f1:.4f}")

# B) Recall-priority
recall_threshold            = 0.5
best_precision_under_recall = -1.0
found_recall_target         = False

for t in thresholds:
    pred = (y_val_proba >= t).astype(int)
    r    = recall_score(y_val2, pred, zero_division=0)
    p    = precision_score(y_val2, pred, zero_division=0)
    if r >= min_recall_target and p > best_precision_under_recall:
        best_precision_under_recall = p
        recall_threshold            = t
        found_recall_target         = True

if found_recall_target:
    print(f"Recall-priority threshold: {recall_threshold:.4f}  |  Val precision: {best_precision_under_recall:.4f}")
else:
    best_val_f2 = -1.0
    for t in thresholds:
        f2 = fbeta_score(y_val2, (y_val_proba >= t).astype(int), beta=2, zero_division=0)
        if f2 > best_val_f2:
            best_val_f2, recall_threshold = f2, t
    print(f"Recall target not reachable; F2-optimised threshold: {recall_threshold:.4f}  |  Val F2: {best_val_f2:.4f}")

# --------------------------------------------------------------
# Evaluate on TEST

def evaluate_and_print(y_true, y_proba, threshold, title):
    y_pred = (y_proba >= threshold).astype(int)
    print(f"\nTEST RESULTS ({title})")
    print(f"Threshold Used: {threshold:.4f}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F2 Score:  {fbeta_score(y_true, y_pred, beta=2, zero_division=0):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

y_pred_proba = final_model.predict_proba(X_test)[:, 1]

evaluate_and_print(y_test, y_pred_proba, best_threshold,   "XGBoost Healthcare (max F1)")
evaluate_and_print(y_test, y_pred_proba, recall_threshold, "XGBoost Healthcare (recall-priority)")

# --------------------------------------------------------------
# Feature importance

print("\n" + "=" * 50)
print("TOP 15 FEATURE IMPORTANCES")
print("=" * 50)
importances = pd.Series(
    final_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

for feat, score in importances.head(15).items():
    print(f"  {feat:<35} {score:.4f}")
