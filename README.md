# Fraud Detection using Machine Learning

A machine learning project that detects automobile insurance fraud using Logistic Regression and Neural Network models built with PyTorch. The project investigates the effect of downsampling the majority class on model performance across three dataset configurations.

---

## Project Structure

```
ML-Project/
├── dataset/
│   ├── fraud_oracle.csv                        # Raw dataset
│   ├── fraud_oracle_preprocessed.csv           # Preprocessed (no downsampling)
│   ├── fraud_oracle_preprocessed_ratio5.csv    # Downsampled 5:1 (valid:fraud)
│   └── fraud_oracle_preprocessed_ratio3.csv    # Downsampled 3:1 (valid:fraud)
├── models/
│   ├── logistic_regression_model.py            # PyTorch Logistic Regression
│   ├── NN_model.py                             # PyTorch Neural Network (2 hidden layers)
│   └── fraud_oracle_model.py                   # Sklearn multi-model baseline (RF, XGBoost, etc.)
├── utils/
│   ├── preprocess_fraud_oracle.py              # Feature engineering & preprocessing
│   ├── downscale.py                            # Majority-class downsampling utility
│   └── get_distribution.py                     # Class distribution extraction
├── results/
│   └── comparison_results.txt                  # Full comparison output
├── run_all_models.py                           # Unified script to train & compare all models
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses the **Vehicle Insurance Fraud Detection** dataset (`fraud_oracle.csv`) containing **15,420 insurance claims** with a binary target `FraudFound_P` indicating whether the claim was fraudulent.

- **Fraud rate:** ~6% (highly imbalanced)
- **Features:** Demographics, policy details, vehicle attributes, claim circumstances

### Preprocessing (`utils/preprocess_fraud_oracle.py`)

- Converts ordinal categorical features (age ranges, price brackets) to numeric midpoints
- Engineers new features: `Driver_Vehicle_Age_Diff`, `Days_Accident_to_Claim`, `Has_Past_Claims`, `Young_Driver`, `High_Value_Vehicle`, etc.
- Encodes binary indicators for `PoliceReportFiled`, `WitnessPresent`, and temporal flags
- Drops redundant original columns after transformation

### Downsampling (`utils/downscale.py`)

To address class imbalance, majority-class samples are randomly downsampled to create balanced datasets:

| Dataset | Valid:Fraud Ratio | Total Samples |
|---------|:-----------------:|:-------------:|
| No Downsampling | ~15:1 | 15,420 |
| 5:1 Ratio | 5:1 | 5,538 |
| 3:1 Ratio | 3:1 | 3,692 |

---

## Models

### Logistic Regression (PyTorch)

- Single linear layer with sigmoid activation
- Weighted BCE loss (`pos_weight` based on class ratio)
- Adam optimizer (lr=0.001)
- Early stopping on validation ROC-AUC (patience=20)
- Decision threshold: 0.42

### Neural Network (PyTorch)

- Architecture: `Input → 64 (BN + ReLU + Dropout) → 32 (BN + ReLU + Dropout) → 1`
- BatchNorm + Dropout (0.2) for regularization
- Weighted BCE loss with Adam optimizer (lr=0.001, weight_decay=1e-4)
- Early stopping on validation ROC-AUC (patience=30)
- Decision threshold: 0.40

### Training Protocol

- **Split:** 72% train / 8% validation / 20% test (stratified)
- **Scaling:** StandardScaler on all features
- **Early stopping:** Monitors validation ROC-AUC, restores best checkpoint

---

## Results

### Comparison Table

| Dataset | Model | Accuracy | Recall | Precision | F1 Score | ROC-AUC |
|---------|-------|:--------:|:------:|:---------:|:--------:|:-------:|
| No Downsampling | Logistic Regression | 0.5973 | 0.9297 | 0.1228 | 0.2169 | 0.7887 |
| No Downsampling | Neural Network | 0.5979 | 0.9405 | 0.1240 | 0.2191 | 0.8037 |
| 5:1 Ratio | Logistic Regression | 0.6137 | 0.9405 | 0.2944 | 0.4485 | 0.7820 |
| 5:1 Ratio | Neural Network | 0.6814 | 0.8162 | 0.3213 | 0.4611 | 0.8022 |
| 3:1 Ratio | Logistic Regression | 0.6333 | 0.9243 | 0.3995 | 0.5579 | 0.7661 |
| 3:1 Ratio | Neural Network | 0.6847 | 0.8486 | 0.4337 | 0.5740 | 0.7944 |

### Confusion Matrices

#### No Downsampling

**Logistic Regression**
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | 1670 | 1229 |
| **Actual 1** | 13 | 172 |

> TN=1670, FP=1229, FN=13, TP=172

**Neural Network**
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | 1670 | 1229 |
| **Actual 1** | 11 | 174 |

> TN=1670, FP=1229, FN=11, TP=174

#### 5:1 Ratio (Downsampled)

**Logistic Regression**
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | 506 | 417 |
| **Actual 1** | 11 | 174 |

> TN=506, FP=417, FN=11, TP=174

**Neural Network**
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | 604 | 319 |
| **Actual 1** | 34 | 151 |

> TN=604, FP=319, FN=34, TP=151

#### 3:1 Ratio (Downsampled)

**Logistic Regression**
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | 297 | 257 |
| **Actual 1** | 14 | 171 |

> TN=297, FP=257, FN=14, TP=171

**Neural Network**
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | 349 | 205 |
| **Actual 1** | 28 | 157 |

> TN=349, FP=205, FN=28, TP=157

### Key Findings

1. **Downsampling significantly improves F1 score.** Without downsampling, both models achieve very high recall (~93-94%) but extremely low precision (~12%), resulting in F1 around 0.22. With 3:1 downsampling, F1 improves to 0.56-0.57 — a **2.6× improvement**.

2. **Neural Network consistently outperforms Logistic Regression** in ROC-AUC across all dataset configurations (0.80 vs 0.79 on full, 0.80 vs 0.78 on 5:1, 0.79 vs 0.77 on 3:1), indicating better overall discriminative ability.

3. **Downsampling trades recall for precision.** The 3:1 ratio Neural Network reduces recall from 0.94 to 0.85 but improves precision from 0.12 to 0.43 — a much better precision-recall balance for practical fraud detection.

4. **More aggressive downsampling (3:1) yields the best F1 scores** for both models, with the Neural Network at 3:1 achieving the best overall F1 of **0.5740**.

5. **ROC-AUC remains relatively stable** across downsampling ratios, suggesting the models maintain their ranking ability regardless of class balance, while threshold-dependent metrics (precision, F1) benefit substantially from rebalancing.

---

## How to Run

### Prerequisites

```bash
pip install torch pandas numpy scikit-learn
```

### Run All Models & Compare

```bash
python run_all_models.py
```

This will:
- Train Logistic Regression and Neural Network on all 3 dataset versions
- Print a comparison table and confusion matrices
- Save detailed results to `results/comparison_results.txt`

### Run Individual Models

```bash
# Logistic Regression (update dataset path in the script first)
python models/logistic_regression_model.py

# Neural Network (update dataset path in the script first)
python models/NN_model.py
```

### Preprocessing Pipeline

```bash
# Step 1: Preprocess raw data
python utils/preprocess_fraud_oracle.py

# Step 2: Generate downsampled datasets
python utils/downscale.py

# Step 3: Check class distributions
python utils/get_distribution.py
```

---

## Requirements

- Python 3.10+
- PyTorch
- pandas
- NumPy
- scikit-learn
