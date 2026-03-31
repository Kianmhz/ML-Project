import os
import itertools
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
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

# Hold out test set (never touched during tuning)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train+Val set: {X_train_full.shape[0]:,} samples ({y_train_full.sum()} fraud)")
print(f"Test set:      {X_test.shape[0]:,} samples ({y_test.sum()} fraud)\n")

# --------------------------------------------------------------
# Model (variable architecture)

class NeuralNetwork(nn.Module):
    def __init__(self, n_features, hidden_sizes, dropout):
        super().__init__()
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# --------------------------------------------------------------
# Single fold training — returns best val F2 across epochs

def train_fold(X_tr, y_tr, X_vl, y_vl, params, patience=30, max_epochs=500):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    dist       = y_tr.value_counts().sort_index()
    pos_weight = torch.tensor([dist[0] / dist[1]], dtype=torch.float32)

    model     = NeuralNetwork(X_tr.shape[1], params['hidden_sizes'], params['dropout'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_t  = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr.to_numpy(), dtype=torch.float32)
    X_vl_t  = torch.tensor(X_vl_s, dtype=torch.float32)
    y_vl_np = y_vl.to_numpy()

    thresholds = np.linspace(0.05, 0.95, 50)
    best_f2    = -1.0
    no_improve = 0

    for _ in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss_fn(model(X_tr_t), y_tr_t).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(model(X_vl_t)).numpy()

        fold_f2 = max(
            fbeta_score(y_vl_np, (proba >= t).astype(int), beta=2, zero_division=0)
            for t in thresholds
        )

        if fold_f2 > best_f2:
            best_f2    = fold_f2
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_f2


# --------------------------------------------------------------
# Hyperparameter grid

param_grid = {
    'hidden_sizes': [(64, 32), (128, 64), (64, 32, 16)],
    'dropout':      [0.2, 0.4],
    'lr':           [0.001, 0.0005],
    'weight_decay': [1e-4, 1e-5],
}

keys   = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))

print(f"Tuning {len(combos)} hyperparameter combinations with 5-fold CV (metric: F2)...\n")

kf           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_mean_f2 = -1.0
best_params  = None

for combo in combos:
    params   = dict(zip(keys, combo))
    fold_f2s = []

    for tr_idx, vl_idx in kf.split(X_train_full, y_train_full):
        X_tr = X_train_full.iloc[tr_idx]
        y_tr = y_train_full.iloc[tr_idx]
        X_vl = X_train_full.iloc[vl_idx]
        y_vl = y_train_full.iloc[vl_idx]
        fold_f2s.append(train_fold(X_tr, y_tr, X_vl, y_vl, params))

    mean_f2 = np.mean(fold_f2s)
    print(f"Params: {params} | Mean F2: {mean_f2:.4f} | Folds: {[f'{f:.4f}' for f in fold_f2s]}")

    if mean_f2 > best_mean_f2:
        best_mean_f2 = mean_f2
        best_params  = params

print(f"\nBest Params:     {best_params}")
print(f"Best CV Mean F2: {best_mean_f2:.4f}\n")

# --------------------------------------------------------------
# Retrain final model on full training data with best params

X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
)

scaler   = StandardScaler()
X_tr2_s  = scaler.fit_transform(X_tr2)
X_val2_s = scaler.transform(X_val2)
X_test_s = scaler.transform(X_test)

dist       = y_tr2.value_counts().sort_index()
pos_weight = torch.tensor([dist[0] / dist[1]], dtype=torch.float32)

final_model = NeuralNetwork(X_train_full.shape[1], best_params['hidden_sizes'], best_params['dropout'])
optimizer   = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
loss_fn     = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

X_tr2_t  = torch.tensor(X_tr2_s,  dtype=torch.float32)
y_tr2_t  = torch.tensor(y_tr2.to_numpy(), dtype=torch.float32)
X_val2_t = torch.tensor(X_val2_s, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
y_val2_np = y_val2.to_numpy()

thresholds     = np.linspace(0.05, 0.95, 50)
max_epochs      = 1000
best_val_f2    = -1.0
best_state     = None
best_threshold = 0.5
patience       = 50
no_improve     = 0

print("Training final model...")
for epoch in range(1, max_epochs + 1):
    final_model.train()
    optimizer.zero_grad()
    loss_fn(final_model(X_tr2_t), y_tr2_t).backward()
    optimizer.step()

    final_model.eval()
    with torch.no_grad():
        val_proba = torch.sigmoid(final_model(X_val2_t)).numpy()

    epoch_f2, epoch_thresh = max(
        ((fbeta_score(y_val2_np, (val_proba >= t).astype(int), beta=2, zero_division=0), t) for t in thresholds),
        key=lambda x: x[0]
    )

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:>4} | Val F2: {epoch_f2:.4f} | Threshold: {epoch_thresh:.2f}")

    if epoch_f2 > best_val_f2:
        best_val_f2    = epoch_f2
        best_threshold = epoch_thresh
        best_state     = {k: v.clone() for k, v in final_model.state_dict().items()}
        no_improve     = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"\nEarly stopping at epoch {epoch}. Best Val F2: {best_val_f2:.4f}")
        break

final_model.load_state_dict(best_state)
final_model.eval()

# --------------------------------------------------------------
# Final evaluation — F2-optimised threshol

with torch.no_grad():
    y_pred_proba = torch.sigmoid(final_model(X_test_t)).numpy()

# A) Max-F2 threshold (already found during training)
y_pred = (y_pred_proba >= best_threshold).astype(int)
print(f"\nTEST RESULTS (Neural Network Healthcare (max F2))")
print(f"Threshold Used: {best_threshold:.4f}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F2 Score:  {fbeta_score(y_test, y_pred, beta=2, zero_division=0):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")