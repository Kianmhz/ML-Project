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
    average_precision_score, precision_recall_curve
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

df = pd.read_csv('/Users/kianmhz/Desktop/ML-Project/fraud_oracle_preprocessed.csv')

print("=" * 70)
print("FRAUD ORACLE - DETECTION MODEL WITH DOWNSAMPLING & THRESHOLD TUNING")
print("=" * 70)
print(f"\nOriginal Dataset Shape: {df.shape}")
print(f"\nOriginal Target Distribution:")
print(df['FraudFound_P'].value_counts())
print(f"\nOriginal Fraud rate: {df['FraudFound_P'].mean()*100:.2f}%")

# ============================================================
# DOWNSAMPLING
# ============================================================

print("\n" + "=" * 70)
print("DOWNSAMPLING MAJORITY CLASS")
print("=" * 70)

# Separate majority and minority classes
df_majority = df[df['FraudFound_P'] == 0]
df_minority = df[df['FraudFound_P'] == 1]

print(f"Majority class (non-fraud): {len(df_majority)}")
print(f"Minority class (fraud): {len(df_minority)}")

# Downsample majority class to different ratios
downsample_ratios = {
    '1:1': len(df_minority),
    '2:1': len(df_minority) * 2,
    '3:1': len(df_minority) * 3,
    '5:1': len(df_minority) * 5
}

print("\nDownsampling ratios to test:")
for ratio, n_samples in downsample_ratios.items():
    print(f"  {ratio} (majority:minority) -> {n_samples} majority samples")

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

# ============================================================
# HELPER FUNCTION: FIND OPTIMAL THRESHOLD
# ============================================================

def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """Find the optimal threshold that maximizes the given metric."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    results = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced':
            # Balance between precision and recall
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'score': score,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        })
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score, pd.DataFrame(results)

# ============================================================
# HELPER FUNCTION: TRAIN AND EVALUATE MODEL
# ============================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name, use_scaled=False, scaler=None, numerical_features=None):
    """Train model and return metrics with threshold tuning."""
    
    if use_scaled and scaler is not None:
        X_train_use = X_train.copy()
        X_test_use = X_test.copy()
        X_train_use[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test_use[numerical_features] = scaler.transform(X_test[numerical_features])
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    model.fit(X_train_use, y_train)
    y_prob = model.predict_proba(X_test_use)[:, 1]
    
    # Default threshold (0.5)
    y_pred_default = model.predict(X_test_use)
    
    # Find optimal threshold
    optimal_threshold, optimal_score, threshold_df = find_optimal_threshold(y_test, y_prob, metric='f1')
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    results = {
        'default_threshold': {
            'threshold': 0.5,
            'accuracy': float(accuracy_score(y_test, y_pred_default)),
            'precision': float(precision_score(y_test, y_pred_default, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_default, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred_default, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred_default).tolist()
        },
        'optimal_threshold': {
            'threshold': float(optimal_threshold),
            'accuracy': float(accuracy_score(y_test, y_pred_optimal)),
            'precision': float(precision_score(y_test, y_pred_optimal, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_optimal, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred_optimal, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred_optimal).tolist()
        },
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'avg_precision': float(average_precision_score(y_test, y_prob)),
        'model': model,
        'y_prob': y_prob
    }
    
    return results

# ============================================================
# EXPERIMENT: DIFFERENT DOWNSAMPLING RATIOS
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTING WITH DIFFERENT DOWNSAMPLING RATIOS")
print("=" * 70)

all_results = {}

for ratio_name, n_majority_samples in downsample_ratios.items():
    print(f"\n{'='*60}")
    print(f"RATIO: {ratio_name}")
    print(f"{'='*60}")
    
    # Downsample majority class
    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=n_majority_samples,
        random_state=42
    )
    
    # Combine minority and downsampled majority
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced dataset size: {len(df_balanced)}")
    print(f"Class distribution:")
    print(df_balanced['FraudFound_P'].value_counts())
    print(f"Fraud rate: {df_balanced['FraudFound_P'].mean()*100:.2f}%")
    
    # Prepare features
    X = df_balanced.drop('FraudFound_P', axis=1)
    y = df_balanced['FraudFound_P']
    
    # Encode categorical features
    for col in categorical_features:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    scaler = StandardScaler()
    
    # Define models
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = (XGBClassifier(
            n_estimators=100, random_state=42, 
            use_label_encoder=False, eval_metric='logloss', n_jobs=-1
        ), False)
    
    ratio_results = {}
    
    for model_name, (model, use_scaled) in models.items():
        print(f"\n  Training {model_name}...")
        
        results = train_and_evaluate(
            X_train, X_test, y_train, y_test, 
            model, model_name, use_scaled, scaler, numerical_features
        )
        
        ratio_results[model_name] = results
        
        # Print results
        default = results['default_threshold']
        optimal = results['optimal_threshold']
        
        print(f"    Default (0.5):  Precision={default['precision']:.4f}, Recall={default['recall']:.4f}, F1={default['f1']:.4f}")
        print(f"    Optimal ({optimal['threshold']:.2f}): Precision={optimal['precision']:.4f}, Recall={optimal['recall']:.4f}, F1={optimal['f1']:.4f}")
        print(f"    ROC-AUC: {results['roc_auc']:.4f}")
    
    all_results[ratio_name] = ratio_results

# ============================================================
# COMPARISON: IMBALANCED VS DOWNSAMPLED (TEST ON ORIGINAL DATA)
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON: TESTING ON ORIGINAL IMBALANCED TEST SET")
print("=" * 70)

# Create a common test set from original data
X_orig = df.drop('FraudFound_P', axis=1)
y_orig = df['FraudFound_P']

# Encode categorical features
for col in categorical_features:
    if col in X_orig.columns:
        le = LabelEncoder()
        X_orig[col] = le.fit_transform(X_orig[col].astype(str))

_, X_test_orig, _, y_test_orig = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
)

print(f"\nOriginal test set: {len(X_test_orig)} samples")
print(f"Fraud cases in test set: {y_test_orig.sum()}")

# Train on best downsampled ratio and test on original
print("\n" + "-" * 70)
print("Training with 3:1 ratio, testing on original imbalanced data")
print("-" * 70)

# Use 3:1 ratio (good balance)
df_majority_3to1 = resample(df_majority, replace=False, n_samples=len(df_minority)*3, random_state=42)
df_balanced_3to1 = pd.concat([df_majority_3to1, df_minority]).sample(frac=1, random_state=42)

X_balanced = df_balanced_3to1.drop('FraudFound_P', axis=1)
y_balanced = df_balanced_3to1['FraudFound_P']

for col in categorical_features:
    if col in X_balanced.columns:
        le = LabelEncoder()
        X_balanced[col] = le.fit_transform(X_balanced[col].astype(str))

# Train on full balanced data, test on original test set
scaler = StandardScaler()

comparison_results = {}

models_comparison = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

if XGBOOST_AVAILABLE:
    models_comparison['XGBoost'] = XGBClassifier(
        n_estimators=100, random_state=42,
        use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )

for model_name, model in models_comparison.items():
    print(f"\n{model_name}:")
    
    # Train on balanced data
    model.fit(X_balanced, y_balanced)
    y_prob = model.predict_proba(X_test_orig)[:, 1]
    
    # Test with different thresholds
    for threshold in [0.5, 0.4, 0.3, 0.25, 0.2]:
        y_pred = (y_prob >= threshold).astype(int)
        
        precision = precision_score(y_test_orig, y_pred, zero_division=0)
        recall = recall_score(y_test_orig, y_pred, zero_division=0)
        f1 = f1_score(y_test_orig, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test_orig, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"  Threshold {threshold:.2f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f} | TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    
    # Find optimal threshold
    optimal_threshold, _, _ = find_optimal_threshold(y_test_orig, y_prob, metric='f1')
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    cm_optimal = confusion_matrix(y_test_orig, y_pred_optimal)
    tn, fp, fn, tp = cm_optimal.ravel()
    
    comparison_results[model_name] = {
        'optimal_threshold': float(optimal_threshold),
        'precision': float(precision_score(y_test_orig, y_pred_optimal, zero_division=0)),
        'recall': float(recall_score(y_test_orig, y_pred_optimal, zero_division=0)),
        'f1': float(f1_score(y_test_orig, y_pred_optimal, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test_orig, y_prob)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }
    
    print(f"  Optimal ({optimal_threshold:.2f}): Precision={comparison_results[model_name]['precision']:.4f}, Recall={comparison_results[model_name]['recall']:.4f}, F1={comparison_results[model_name]['f1']:.4f}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY: BEST RESULTS BY DOWNSAMPLING RATIO")
print("=" * 70)

summary_data = []
for ratio_name, ratio_results in all_results.items():
    for model_name, results in ratio_results.items():
        if model_name == 'model' or model_name == 'y_prob':
            continue
        summary_data.append({
            'Ratio': ratio_name,
            'Model': model_name,
            'Default_F1': results['default_threshold']['f1'],
            'Optimal_Threshold': results['optimal_threshold']['threshold'],
            'Optimal_F1': results['optimal_threshold']['f1'],
            'Optimal_Recall': results['optimal_threshold']['recall'],
            'ROC_AUC': results['roc_auc']
        })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Best overall
best_idx = summary_df['Optimal_F1'].idxmax()
best_config = summary_df.loc[best_idx]
print(f"\n{'='*70}")
print("BEST CONFIGURATION:")
print(f"  Ratio: {best_config['Ratio']}")
print(f"  Model: {best_config['Model']}")
print(f"  Optimal Threshold: {best_config['Optimal_Threshold']:.2f}")
print(f"  F1-Score: {best_config['Optimal_F1']:.4f}")
print(f"  Recall: {best_config['Optimal_Recall']:.4f}")
print(f"  ROC-AUC: {best_config['ROC_AUC']:.4f}")
print(f"{'='*70}")

# ============================================================
# DETAILED CONFUSION MATRIX AND SAMPLE COUNTS
# ============================================================

print("\n" + "=" * 70)
print("DETAILED CONFUSION MATRIX & SAMPLE COUNTS")
print("=" * 70)

print("\n" + "-" * 70)
print("SAMPLE COUNTS BY DOWNSAMPLING RATIO")
print("-" * 70)
print(f"\nOriginal Dataset:")
print(f"  Total samples: {len(df):,}")
print(f"  Non-Fraud (0): {len(df_majority):,}")
print(f"  Fraud (1): {len(df_minority):,}")
print(f"  Fraud Rate: {len(df_minority)/len(df)*100:.2f}%")

for ratio_name, n_majority_samples in downsample_ratios.items():
    total = n_majority_samples + len(df_minority)
    fraud_rate = len(df_minority) / total * 100
    print(f"\n{ratio_name} Ratio:")
    print(f"  Total samples: {total:,}")
    print(f"  Non-Fraud (0): {n_majority_samples:,}")
    print(f"  Fraud (1): {len(df_minority):,}")
    print(f"  Fraud Rate: {fraud_rate:.2f}%")

print("\n" + "-" * 70)
print("CONFUSION MATRICES BY RATIO AND MODEL (OPTIMAL THRESHOLD)")
print("-" * 70)

for ratio_name, ratio_results in all_results.items():
    print(f"\n{'='*50}")
    print(f"RATIO: {ratio_name}")
    print(f"{'='*50}")
    
    for model_name, results in ratio_results.items():
        if model_name in ['model', 'y_prob']:
            continue
            
        opt = results['optimal_threshold']
        cm = opt['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        print(f"\n{model_name} (Threshold: {opt['threshold']:.2f}):")
        print(f"  Confusion Matrix:")
        print(f"                    Predicted")
        print(f"                 Non-Fraud  Fraud")
        print(f"  Actual Non-Fraud  {tn:>6}   {fp:>6}")
        print(f"  Actual Fraud      {fn:>6}   {tp:>6}")
        print(f"  ")
        print(f"  Metrics:")
        print(f"    True Positives (TP):  {tp:>6} (Frauds correctly detected)")
        print(f"    True Negatives (TN):  {tn:>6} (Non-frauds correctly identified)")
        print(f"    False Positives (FP): {fp:>6} (False alarms)")
        print(f"    False Negatives (FN): {fn:>6} (Missed frauds)")
        print(f"    Precision: {opt['precision']:.4f}")
        print(f"    Recall:    {opt['recall']:.4f}")
        print(f"    F1-Score:  {opt['f1']:.4f}")

print("\n" + "-" * 70)
print("CONFUSION MATRICES ON ORIGINAL TEST SET (3:1 DOWNSAMPLED TRAINING)")
print("-" * 70)

for model_name, results in comparison_results.items():
    cm = results['confusion_matrix']
    
    print(f"\n{model_name} (Threshold: {results['optimal_threshold']:.2f}):")
    print(f"  Test Set Size: {len(y_test_orig):,} samples")
    print(f"  Fraud cases in test: {y_test_orig.sum():,}")
    print(f"  Non-fraud cases in test: {(y_test_orig == 0).sum():,}")
    print(f"  ")
    print(f"  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Non-Fraud  Fraud")
    print(f"  Actual Non-Fraud  {cm['tn']:>6}   {cm['fp']:>6}")
    print(f"  Actual Fraud      {cm['fn']:>6}   {cm['tp']:>6}")
    print(f"  ")
    print(f"  Detailed Breakdown:")
    print(f"    True Positives (TP):  {cm['tp']:>6} (Frauds correctly detected)")
    print(f"    True Negatives (TN):  {cm['tn']:>6} (Non-frauds correctly identified)")
    print(f"    False Positives (FP): {cm['fp']:>6} (False alarms)")
    print(f"    False Negatives (FN): {cm['fn']:>6} (Missed frauds)")
    print(f"    ")
    print(f"    Detection Rate: {cm['tp']}/{cm['tp']+cm['fn']} = {cm['tp']/(cm['tp']+cm['fn'])*100:.1f}% of frauds caught")
    print(f"    False Alarm Rate: {cm['fp']}/{cm['fp']+cm['tn']} = {cm['fp']/(cm['fp']+cm['tn'])*100:.2f}%")
    print(f"    ")
    print(f"  Performance Metrics:")
    print(f"    Precision: {results['precision']:.4f}")
    print(f"    Recall:    {results['recall']:.4f}")
    print(f"    F1-Score:  {results['f1']:.4f}")
    print(f"    ROC-AUC:   {results['roc_auc']:.4f}")

# ============================================================
# SAVE RESULTS
# ============================================================

# Clean results for JSON (remove non-serializable objects)
clean_results = {}
for ratio_name, ratio_results in all_results.items():
    clean_results[ratio_name] = {}
    for model_name, results in ratio_results.items():
        clean_results[ratio_name][model_name] = {
            k: v for k, v in results.items() 
            if k not in ['model', 'y_prob']
        }

output = {
    'downsampling_experiments': clean_results,
    'comparison_on_original_test': comparison_results,
    'best_configuration': {
        'ratio': best_config['Ratio'],
        'model': best_config['Model'],
        'optimal_threshold': float(best_config['Optimal_Threshold']),
        'f1_score': float(best_config['Optimal_F1']),
        'recall': float(best_config['Optimal_Recall']),
        'roc_auc': float(best_config['ROC_AUC'])
    }
}

with open('/Users/kianmhz/Desktop/ML-Project/results_oracle_downsampled.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults saved to results_oracle_downsampled.json")
