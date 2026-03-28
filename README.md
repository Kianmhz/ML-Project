# Fraud Detection using Machine Learning

A machine learning project comparing supervised and unsupervised fraud detection across two datasets: automobile insurance claims (Fraud Oracle) and Medicare provider billing (Healthcare Fraud). Models include Logistic Regression, Neural Network, XGBoost, and Isolation Forest.

---

## Project Structure

```
ML-Project/
├── dataset/
│   ├── fraud_oracle.csv                         # Raw Oracle dataset
│   ├── fraud_oracle_preprocessed.csv            # Preprocessed Oracle dataset
│   └── healthcare_fraud/
│       ├── Train-1542865627584.csv              # Provider fraud labels
│       ├── Train_Beneficiarydata-1542865627584.csv
│       ├── Train_Inpatientdata-1542865627584.csv
│       └── Train_Outpatientdata-1542865627584.csv
│   └── healthcare_fraud_preprocessed.csv        # Aggregated provider-level dataset
├── models/
│   ├── logistic_regression_model.py             # Oracle — Logistic Regression
│   ├── NN_model.py                              # Oracle — Neural Network
│   ├── xgboost_model.py                         # Oracle — XGBoost (k-fold tuned)
│   ├── unsupervised_model.py                    # Oracle — Isolation Forest + Autoencoder
│   ├── logistic_regression_healthcare.py        # Healthcare — Logistic Regression
│   ├── NN_healthcare.py                         # Healthcare — Neural Network (k-fold tuned)
│   ├── xgboost_healthcare.py                    # Healthcare — XGBoost (k-fold tuned)
│   └── unsupervised_healthcare.py               # Healthcare — Isolation Forest
├── utils/
│   ├── preprocess_fraud_oracle.py               # Oracle feature engineering
│   ├── preprocess_healthcare_fraud.py           # Healthcare join + aggregation pipeline
│   ├── downscale.py                             # Majority-class downsampling utility
│   └── get_distribution.py                      # Class distribution extraction
├── json/
│   ├── distribution_oracle.json
│   └── distribution_healthcare.json
├── fraud_oracle_results.txt                     # Full Oracle results for all models
├── run_oracle_results.py                        # Runs all Oracle models, saves results
├── requirements.txt
└── README.md
```

---

## Datasets

### 1. Fraud Oracle — Vehicle Insurance Claims

**Source:** Vehicle Insurance Fraud Detection dataset
**Rows:** 15,420 insurance claims | **Target:** `FraudFound_P` (binary)
**Fraud rate:** ~6% (highly imbalanced)

**Preprocessing** (`utils/preprocess_fraud_oracle.py`):
- Ordinal string ranges (age, price, vehicle age) mapped to numeric midpoints
- Cyclical sin/cos encoding for month and day-of-week features
- Derived features: `Driver_Vehicle_Age_Diff`, `Days_Accident_to_Claim`, `Claim_Before_Accident`, `Has_Past_Claims`, `Young_Driver`, `High_Value_Vehicle`
- Binary encoding for `PoliceReportFiled`, `WitnessPresent`
- Rep-level claim count (`Rep_Claim_Count`) as a network frequency feature
- Redundant binary flags removed (duplicates of categorical columns)

### 2. Healthcare Fraud — Medicare Provider Billing

**Source:** Medicare Provider Fraud Detection dataset (Kaggle)
**Structure:** 4 relational CSV files joined by foreign keys

| File | Rows | Description |
|------|-----:|-------------|
| Train labels | 5,410 | Provider ID + `PotentialFraud` (Yes/No) |
| Beneficiary data | 138,556 | Patient demographics, chronic conditions |
| Inpatient claims | 40,474 | Hospital admissions with diagnosis/procedure codes |
| Outpatient claims | 517,737 | Outpatient visits with diagnosis codes |

**Fraud rate:** ~9.4% | **Target:** Provider-level fraud (one row per provider)

**Preprocessing** (`utils/preprocess_healthcare_fraud.py`):

The multi-file structure is handled as a relational join pipeline:
1. **Beneficiary engineering** — compute patient age from DOB, recode chronic conditions (1/2 → 1/0), flag deceased patients
2. **Claim-level engineering** — compute `ClaimDuration`, `HospitalStay`, `DiagnosisCount`, `ProcedureCount`, `SamePhysician` (self-referral flag)
3. **JOIN** — attach beneficiary demographics to each claim via `BeneID`
4. **GROUP BY Provider** — aggregate all claims into one row per provider with 40 features

Key aggregated features include:
- Volume: `total_claims`, `ip_ratio`, `claims_per_bene`, `unique_benes`
- Billing: `total_reimbursed`, `avg_claim_reimbursed`, `max_claim_reimbursed`
- Temporal: `avg_claim_duration`, `avg_hospital_stay`, `max_hospital_stay`
- Network: `unique_attending`, `pct_same_physician`, `physicians_per_bene`
- Patient profile: `avg_bene_age`, `avg_chronic_count`, `pct_deceased`, per-condition prevalence rates

---

## Models

### Supervised

| Model | Key Details |
|-------|-------------|
| **Logistic Regression** | PyTorch linear layer, weighted BCE loss, Adam optimizer, early stopping on val ROC-AUC |
| **Neural Network** | Variable architecture (k-fold tuned), BatchNorm + Dropout, weighted BCE, early stopping on val F1 |
| **XGBoost** | `binary:logistic`, `scale_pos_weight`, k-fold hyperparameter tuning optimised for F1, two evaluation thresholds |

### Unsupervised

| Model | Key Details |
|-------|-------------|
| **Isolation Forest** | Trained on non-fraud samples only, `contamination` set to dataset fraud rate, anomaly score = `-score_samples` |
| **Autoencoder** | PyTorch encoder-decoder trained on non-fraud; reconstruction error = anomaly score (Oracle only — underperformed on healthcare) |

### Training Protocol

- **Split:** 80% train+val / 20% test (stratified), held-out test never used during tuning
- **Scaling:** `StandardScaler` fit on training fold only, applied to val/test
- **Class imbalance:** `pos_weight = n_neg / n_pos` in BCE loss; `scale_pos_weight` in XGBoost
- **Thresholds:** Two thresholds evaluated — max F1 and recall-priority (≥0.91 recall, max precision)
- **Target encoding** (Oracle XGBoost): `Rep_Fraud_Rate`, `Make_Fraud_Rate`, `PolicyType_Fraud_Rate` computed on training fold only inside each CV fold to prevent leakage

---

## Results

### Fraud Oracle Dataset

| Model | ROC AUC | F1 (max) | Recall (recall-priority) |
|-------|:-------:|:--------:|:------------------------:|
| Logistic Regression | 0.783 | 0.212 | 0.935 |
| Neural Network | 0.810 | 0.274 | — |
| XGBoost | **0.848** | **0.284** | **0.946** |
| Isolation Forest (unsupervised) | 0.507 | 0.121 | 0.995 |

### Healthcare Fraud Dataset

| Model | ROC AUC | F1 (max) | Recall (recall-priority) |
|-------|:-------:|:--------:|:------------------------:|
| Logistic Regression | 0.958 | 0.654 | 0.921 |
| Neural Network | 0.965 | 0.673 | 0.911 |
| XGBoost | **0.970** | **0.711** | **0.980** |
| Isolation Forest (unsupervised) | 0.891 | 0.485 | 0.911 |

### Key Findings

1. **Dataset quality dominates model choice.** Switching from Fraud Oracle to the Healthcare dataset improved XGBoost ROC AUC from 0.848 → 0.970 and F1 from 0.28 → 0.71 — a larger gain than any model or tuning improvement applied to Oracle.

2. **Fraud Oracle hit a structural ceiling.** All supervised models clustered between 0.78–0.85 ROC AUC regardless of architecture or tuning. The dataset lacks linking IDs across claims, preventing temporal and network feature engineering.

3. **XGBoost consistently outperforms NNs on tabular data.** Tree models are optimised for row-level feature thresholds, which is how fraud manifests in structured records. The NN gap was smaller on the healthcare dataset (0.005 ROC AUC) than on Oracle (0.038), as richer features reduce the NN's relative disadvantage.

4. **Unsupervised methods reflect the data's anomaly structure.** On Oracle, Isolation Forest ROC AUC was 0.507 (random) — fraud cases are not statistical outliers. On Healthcare, it achieved 0.891 — fraudulent providers genuinely bill anomalously. The unsupervised performance gap directly measures how "detectable without labels" the fraud is.

5. **Feature engineering from relational joins is the highest-leverage action.** `total_reimbursed` alone accounted for 30.8% of XGBoost feature importance on the healthcare dataset. This feature required joining 4 CSV files and aggregating 558k claims — it cannot be derived from a flat single-table dataset.

6. **Class weights matter less when fraud is easier to separate.** On Healthcare (9.4% fraud rate, strong billing signals), removing `pos_weight` from Logistic Regression changed ROC AUC by only +0.003. On Oracle (6% fraud rate, weak signals), class weighting had a more meaningful impact.

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Fraud Oracle Pipeline

```bash
# Step 1: Preprocess raw data
python utils/preprocess_fraud_oracle.py

# Step 2: Run individual models
python models/logistic_regression_model.py
python models/NN_model.py
python models/xgboost_model.py
python models/unsupervised_model.py

# Step 3: Run all Oracle models and save results
python run_oracle_results.py
```

### Healthcare Fraud Pipeline

```bash
# Step 1: Preprocess — joins 4 CSV files and aggregates to provider level
python utils/preprocess_healthcare_fraud.py

# Step 2: Run individual models
python models/logistic_regression_healthcare.py
python models/NN_healthcare.py
python models/xgboost_healthcare.py
python models/unsupervised_healthcare.py
```

---

## Requirements

- Python 3.10+
- PyTorch
- pandas
- NumPy
- scikit-learn
- xgboost
