import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Load class distribution and compute class weights
with open('/Users/kianmhz/Desktop/ML-Project/distribution_oracle.json', 'r') as f:
    distribution = json.load(f)

# Compute class weights: total_samples / (n_classes * class_count)
total_samples = sum(distribution.values())
n_classes = len(distribution)
class_weights = {int(k): total_samples / (n_classes * v) for k, v in distribution.items()}
print(f"Class weights from distribution_oracle.json: {class_weights}")

# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

df = pd.read_csv('/Users/kianmhz/Desktop/ML-Project/fraud_oracle_preprocessed.csv')

print("=" * 60)
print("FRAUD ORACLE - DETECTION MODEL")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")
print(f"\nTarget Distribution:")
print(df['FraudFound_P'].value_counts())
print(f"\nFraud rate: {df['FraudFound_P'].mean()*100:.2f}%")

# ============================================================
# DEFINE FEATURES
# ============================================================

numerical_features = [
    'Age_Numeric', 'PolicyHolderAge_Numeric', 'VehiclePrice_Numeric', 'VehicleAge_Numeric',
    'DaysPolicyAccident_Numeric', 'DaysPolicyClaim_Numeric', 'PastClaims_Numeric',
    'NumSupplements_Numeric', 'AddressChangeClaim_Numeric', 'NumCars_Numeric',
    'Deductible', 'DriverRating', 'WeekOfMonth', 'WeekOfMonthClaimed', 'Year',
    'Driver_Vehicle_Age_Diff', 'PolicyHolder_Driver_Age_Diff', 'Days_Accident_to_Claim',
    'Month_Numeric', 'MonthClaimed_Numeric', 'DayOfWeek_Numeric', 'DayOfWeekClaimed_Numeric',
    'PoliceReportFiled_Binary', 'WitnessPresent_Binary',
    'Has_Past_Claims', 'New_Vehicle', 'Young_Driver', 'High_Value_Vehicle',
    'External_Agent', 'Is_Urban', 'Policy_Holder_Fault', 'Accident_Weekend', 'Claim_Weekend'
]

categorical_features = [
    'Make', 'AccidentArea', 'Sex', 'MaritalStatus', 'Fault',
    'PolicyType', 'VehicleCategory', 'AgentType', 'BasePolicy'
]

# Filter to only existing columns
numerical_features = [f for f in numerical_features if f in df.columns]
categorical_features = [f for f in categorical_features if f in df.columns]

print(f"\nNumerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# ============================================================
# PREPROCESSING
# ============================================================

X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

# ============================================================
# MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(
        class_weight=class_weights, max_iter=1000, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, class_weight=class_weights, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    'HistGradient Boosting': HistGradientBoostingClassifier(
        max_iter=100, random_state=42
    )
}

if XGBOOST_AVAILABLE:
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models['XGBoost'] = XGBClassifier(
        n_estimators=100, scale_pos_weight=scale_pos_weight,
        random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for Logistic Regression, unscaled for tree-based
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'model': model
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Avg Precision: {avg_precision:.4f}")

# ============================================================
# BEST MODEL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("BEST MODEL ANALYSIS")
print("=" * 60)

# Find best model by F1-score
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

print(f"\nBest Model: {best_model_name}")
print(f"F1-Score: {results[best_model_name]['f1']:.4f}")

# Get predictions
if best_model_name == 'Logistic Regression':
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)

rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# ============================================================
# CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("CROSS-VALIDATION (5-Fold)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name in ['Random Forest', 'XGBoost'] if XGBOOST_AVAILABLE else ['Random Forest']:
    model = results[name]['model']
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(f"\n{name}:")
    print(f"  CV F1 Scores: {cv_scores}")
    print(f"  Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ============================================================
# SAVE RESULTS
# ============================================================

results_summary = {
    name: {k: v for k, v in metrics.items() if k != 'model'}
    for name, metrics in results.items()
}

with open('/Users/kianmhz/Desktop/ML-Project/results_oracle.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "=" * 60)
print("Results saved to results_oracle.json")
print("=" * 60)
