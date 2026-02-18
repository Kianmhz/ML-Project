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
categorical_features = df.select_dtypes(include=['object','category']).columns

df.head()

# Seperate Data into input features and expected outputs
X = df.drop('FraudFound_P', axis=1).copy()
y = df['FraudFound_P'].astype(int)

# Convert categorical features to numerical
# (One-Hot Encoding instead of LabelEncoder)
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# --------------------------------------------------------------

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale numerical features to help with training
scaler = StandardScaler()

# Only scale feature columns (exclude target)
# (after one-hot, everything is numeric; scaling the full matrix is fine)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------

# Defining model
class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(1)

# Training Model
model = LogisticRegression(n_features=X_train.shape[1])

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert training data to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)

for epoch in range(20):
    model.train()
    outputs = model(X_train_tensor)
    optimizer.zero_grad()
    loss = loss_function(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()

with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
    y_pred_proba = model(X_test_tensor).cpu().numpy()
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{confusion}")
