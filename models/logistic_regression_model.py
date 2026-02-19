import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

# Importing Dataset
df = pd.read_csv('/Users/kianmhz/Desktop/ML-Project/dataset/fraud_oracle_preprocessed.csv')

# --------------------------------------------------------------
# Seperating Numerical and Categorical features

numerical_features = df.select_dtypes(include=[np.number])
categorical_features = df.select_dtypes(include=['object', 'category']).columns

# Seperate Data into input features and expected outputs
X = df.drop('FraudFound_P', axis=1).copy()
y = df['FraudFound_P'].astype(int)

# One-hot encode categorical features
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# --------------------------------------------------------------
# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train-validation split (10% of training)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples\n")

# --------------------------------------------------------------
# Scale features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

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

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

# --------------------------------------------------------------
# Weighted BCE (per-sample weights)

distribution = y_train.value_counts().sort_index()
pos_weight = torch.tensor([distribution[0] / distribution[1]], dtype=torch.float32)

loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------------------
# Early stopping settings (ROC-AUC on validation)

max_epochs = 1000
patience = 20
best_val_roc = -float("inf")
best_epoch = -1
epochs_no_improve = 0
best_state_dict = None

for epoch in range(1, max_epochs + 1):
    # ---- train ----
    model.train()
    outputs = model(X_train_tensor)

    optimizer.zero_grad()
    loss = loss_function(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # ---- validate each epoch ----
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_proba = torch.sigmoid(val_logits).cpu().numpy()
        val_roc = roc_auc_score(y_val, val_proba)

        try:
            val_roc = roc_auc_score(y_val, val_proba)
        except ValueError:
            val_roc = float("nan")

    # print progress occasionally (or every epoch if you want)
    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:>3} | Train Loss: {loss.item():.6f} | Val ROC-AUC: {val_roc:.6f}")

    # ---- early stopping logic (stagnation) ----
    improved = (not np.isnan(val_roc)) and (val_roc > best_val_roc)

    if improved:
        best_val_roc = val_roc
        best_epoch = epoch
        epochs_no_improve = 0
        best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"\nEarly stopping: no Val ROC-AUC improvement for {patience} epochs.")
        print(f"Best Val ROC-AUC: {best_val_roc:.6f} at epoch {best_epoch}")
        break

# --------------------------------------------------------------
# Restore best model (best validation ROC-AUC), then test ONCE

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

model.eval()
with torch.no_grad():
    test_logits = model(X_test_tensor)
    y_pred_proba = torch.sigmoid(test_logits).cpu().numpy()
    y_pred = (y_pred_proba >= 0.42).astype(int)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)
confusion = confusion_matrix(y_test, y_pred)

print(f"\nTEST RESULTS (using best val ROC-AUC model from epoch {best_epoch})")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"\nConfusion Matrix:\n{confusion}")