import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
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

# --------------------------------------------------------------
# Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"Training set:   {X_train.shape[0]:,} samples ({y_train.sum()} fraud)")
print(f"Validation set: {X_val.shape[0]:,} samples ({y_val.sum()} fraud)")
print(f"Test set:       {X_test.shape[0]:,} samples ({y_test.sum()} fraud)\n")

# --------------------------------------------------------------
# Scale

scaler        = StandardScaler()
X_train_s     = scaler.fit_transform(X_train)
X_val_s       = scaler.transform(X_val)
X_test_s      = scaler.transform(X_test)

# --------------------------------------------------------------
# Model

class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)

model = LogisticRegression(n_features=X_train.shape[1])

# --------------------------------------------------------------
# Tensors

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
X_val_t   = torch.tensor(X_val_s,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32)

# --------------------------------------------------------------
# Weighted BCE + optimizer

distribution = y_train.value_counts().sort_index()
pos_weight   = torch.tensor([distribution[0] / distribution[1]], dtype=torch.float32)
loss_fn      = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer    = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------------------
# Training with early stopping on val ROC-AUC

max_epochs      = 1000
patience        = 50
best_val_roc    = -float('inf')
best_epoch      = -1
epochs_no_impr  = 0
best_state      = None

for epoch in range(1, max_epochs + 1):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_proba = torch.sigmoid(model(X_val_t)).numpy()
        try:
            val_roc = roc_auc_score(y_val, val_proba)
        except ValueError:
            val_roc = float('nan')

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:>4} | Loss: {loss.item():.6f} | Val ROC-AUC: {val_roc:.6f}")

    if (not np.isnan(val_roc)) and val_roc > best_val_roc:
        best_val_roc   = val_roc
        best_epoch     = epoch
        epochs_no_impr = 0
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_impr += 1

    if epochs_no_impr >= patience:
        print(f"\nEarly stopping at epoch {epoch}. Best Val ROC-AUC: {best_val_roc:.6f} (epoch {best_epoch})")
        break

if best_state:
    model.load_state_dict(best_state)
model.eval()

# --------------------------------------------------------------
# Threshold optimisation on val set

with torch.no_grad():
    val_proba = torch.sigmoid(model(X_val_t)).numpy()

thresholds        = np.linspace(0.05, 0.95, 50)

# A) Max-F1
best_threshold = 0.5
best_val_f2    = -1.0
for t in thresholds:
    f = fbeta_score(y_val, (val_proba >= t).astype(int), beta=2, zero_division=0)
    if f > best_val_f2:
        best_val_f2, best_threshold = f, t

print(f"\nBest threshold (max F2):  {best_threshold:.4f}  |  Val F2: {best_val_f2:.4f}")

# --------------------------------------------------------------
# Evaluate on TEST

with torch.no_grad():
    y_pred_proba = torch.sigmoid(model(X_test_t)).numpy()

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

evaluate_and_print(y_test, y_pred_proba, best_threshold,   "Logistic Regression Healthcare (max F2)")
