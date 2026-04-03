"""
visualize.py — Generate report figures for the fraud detection project.

Saves 4 figures to figures/ that are NOT already covered by the UI screenshots:
  figures/roc_curves.png           — ROC curves, all models, Healthcare vs Oracle (side-by-side)
  figures/pr_curves.png            — Precision-Recall curves, Healthcare
  figures/feature_importance.png   — XGBoost feature importance, Healthcare top-20
  figures/threshold_sensitivity.png — F2 / Recall / Precision vs threshold, XGBoost Healthcare

Already covered by UI screenshots (skipped here):
  UI_images/CM.png         — Confusion matrices, all 4 Healthcare models
  UI_images/all_models.png — Grouped bar chart: ROC AUC, F2, Recall, Precision, F1
  UI_images/cards.png      — Per-model metric cards with thresholds
  UI_images/fraud_rate.png — Class distribution donut charts (both datasets)

Oracle ROC curves use sklearn LogisticRegression + XGBClassifier (not the original
PyTorch LR). AUC values may differ slightly from fraud_oracle_results.txt.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, fbeta_score, recall_score, precision_score,
)
from xgboost import XGBClassifier

# ──────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR  = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    'XGBoost':          '#E07B39',
    'Neural Network':   '#4B7BB5',
    'Logistic Reg.':    '#5A9E6F',
    'Isolation Forest': '#9C6BB5',
}

plt.rcParams.update({
    'font.family':      'sans-serif',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'figure.dpi':       150,
})

def save(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/{name}")

# ──────────────────────────────────────────────────────────────
# Healthcare data
# ──────────────────────────────────────────────────────────────

print("\n[1/2] Loading Healthcare data...")
hc_df = pd.read_csv(os.path.join(BASE_DIR, 'dataset', 'healthcare_fraud_preprocessed.csv'))
X_hc  = hc_df.drop(columns=['Provider', 'PotentialFraud'])
y_hc  = hc_df['PotentialFraud'].astype(int)

# No stratify — matches the original IsolationForest.py split so all models
# share the same test set and the IF AUC aligns with the reported value (0.7918).
X_hc_train, X_hc_test, y_hc_train, y_hc_test = train_test_split(
    X_hc, y_hc, test_size=0.2, random_state=42
)

# ──────────────────────────────────────────────────────────────
# Train Healthcare models to obtain score arrays
# ──────────────────────────────────────────────────────────────

# --- XGBoost (Healthcare) — best params from 5-fold CV ---
print("  Training XGBoost (Healthcare)...")

scaler_xgb = StandardScaler()
X_hc_tr, X_hc_val, y_hc_tr, y_hc_val = train_test_split(
    X_hc_train, y_hc_train, test_size=0.1, random_state=42
)
scaler_xgb.fit(X_hc_tr)

dist    = y_hc_tr.value_counts().sort_index()
pos_w   = dist[0] / dist[1]

xgb_hc = XGBClassifier(
    objective='binary:logistic', eval_metric='auc',
    max_depth=6, learning_rate=0.05, subsample=0.9,
    min_child_weight=3, reg_lambda=3.0,
    n_estimators=1000, early_stopping_rounds=50,
    scale_pos_weight=pos_w,
    random_state=42, tree_method='hist', verbosity=0,
)
xgb_hc.fit(
    scaler_xgb.transform(X_hc_tr), y_hc_tr,
    eval_set=[(scaler_xgb.transform(X_hc_val), y_hc_val)],
    verbose=False,
)
xgb_hc_scores = xgb_hc.predict_proba(scaler_xgb.transform(X_hc_test))[:, 1]
print(f"    AUC = {roc_auc_score(y_hc_test, xgb_hc_scores):.4f}  (reported: 0.9703)")

# --- Logistic Regression (Healthcare) — PyTorch linear model ---
print("  Training Logistic Regression (Healthcare)...")

class LRModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = nn.Linear(n, 1)
    def forward(self, x):
        return self.linear(x).squeeze(1)

scaler_lr = StandardScaler()
X_lr_tr_s  = scaler_lr.fit_transform(X_hc_tr)
X_lr_val_s = scaler_lr.transform(X_hc_val)
X_lr_test_s = scaler_lr.transform(X_hc_test)

lr_dist   = y_hc_tr.value_counts().sort_index()
lr_pw     = torch.tensor([lr_dist[0] / lr_dist[1]], dtype=torch.float32)
lr_model  = LRModel(X_hc.shape[1])
lr_loss   = nn.BCEWithLogitsLoss(pos_weight=lr_pw)
lr_opt    = torch.optim.Adam(lr_model.parameters(), lr=0.001)

X_lr_tr_t  = torch.tensor(X_lr_tr_s,  dtype=torch.float32)
y_lr_tr_t  = torch.tensor(y_hc_tr.to_numpy(), dtype=torch.float32)
X_lr_val_t = torch.tensor(X_lr_val_s, dtype=torch.float32)
X_lr_test_t = torch.tensor(X_lr_test_s, dtype=torch.float32)

best_roc, best_state_lr, no_impr = -np.inf, None, 0
for _ in range(1000):
    lr_model.train();  lr_opt.zero_grad()
    lr_loss(lr_model(X_lr_tr_t), y_lr_tr_t).backward();  lr_opt.step()
    lr_model.eval()
    with torch.no_grad():
        vp = torch.sigmoid(lr_model(X_lr_val_t)).numpy()
    try:
        vr = roc_auc_score(y_hc_val, vp)
    except ValueError:
        vr = float('nan')
    if not np.isnan(vr) and vr > best_roc:
        best_roc = vr
        best_state_lr = {k: v.clone() for k, v in lr_model.state_dict().items()}
        no_impr = 0
    else:
        no_impr += 1
    if no_impr >= 50:
        break

lr_model.load_state_dict(best_state_lr); lr_model.eval()
with torch.no_grad():
    lr_hc_scores = torch.sigmoid(lr_model(X_lr_test_t)).numpy()
print(f"    AUC = {roc_auc_score(y_hc_test, lr_hc_scores):.4f}  (reported: 0.9618)")

# --- Neural Network (Healthcare) — best params from 5-fold CV ---
print("  Training Neural Network (Healthcare)...")

class NNModel(nn.Module):
    def __init__(self, n, hidden_sizes, dropout):
        super().__init__()
        layers, in_size = [], n
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(1)

scaler_nn  = StandardScaler()
X_nn_tr_s  = scaler_nn.fit_transform(X_hc_tr)
X_nn_val_s = scaler_nn.transform(X_hc_val)
X_nn_test_s = scaler_nn.transform(X_hc_test)

nn_dist  = y_hc_tr.value_counts().sort_index()
nn_pw    = torch.tensor([nn_dist[0] / nn_dist[1]], dtype=torch.float32)
nn_model = NNModel(X_hc.shape[1], hidden_sizes=(64, 32, 16), dropout=0.2)
nn_loss  = nn.BCEWithLogitsLoss(pos_weight=nn_pw)
nn_opt   = torch.optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-4)

X_nn_tr_t  = torch.tensor(X_nn_tr_s,  dtype=torch.float32)
y_nn_tr_t  = torch.tensor(y_hc_tr.to_numpy(), dtype=torch.float32)
X_nn_val_t = torch.tensor(X_nn_val_s, dtype=torch.float32)
X_nn_test_t = torch.tensor(X_nn_test_s, dtype=torch.float32)
y_nn_val_np = y_hc_val.to_numpy()
thresholds  = np.linspace(0.05, 0.95, 50)

best_f2, best_state_nn, no_impr = -1.0, None, 0
for _ in range(1000):
    nn_model.train();  nn_opt.zero_grad()
    nn_loss(nn_model(X_nn_tr_t), y_nn_tr_t).backward();  nn_opt.step()
    nn_model.eval()
    with torch.no_grad():
        vp = torch.sigmoid(nn_model(X_nn_val_t)).numpy()
    f2 = max(fbeta_score(y_nn_val_np, (vp >= t).astype(int), beta=2, zero_division=0) for t in thresholds)
    if f2 > best_f2:
        best_f2 = f2
        best_state_nn = {k: v.clone() for k, v in nn_model.state_dict().items()}
        no_impr = 0
    else:
        no_impr += 1
    if no_impr >= 50:
        break

nn_model.load_state_dict(best_state_nn); nn_model.eval()
with torch.no_grad():
    nn_hc_scores = torch.sigmoid(nn_model(X_nn_test_t)).numpy()
print(f"    AUC = {roc_auc_score(y_hc_test, nn_hc_scores):.4f}  (reported: 0.9636)")

# --- Isolation Forest (Healthcare) ---
print("  Training Isolation Forest (Healthcare)...")
scaler_iso  = StandardScaler()
X_iso_tr_s  = scaler_iso.fit_transform(X_hc_train)
X_iso_test_s = scaler_iso.transform(X_hc_test)

iso_hc = IsolationForest(n_estimators=200, contamination=0.35, random_state=42, n_jobs=-1)
iso_hc.fit(X_iso_tr_s)
iso_hc_scores = -iso_hc.score_samples(X_iso_test_s)
print(f"    AUC = {roc_auc_score(y_hc_test, iso_hc_scores):.4f}  (reported: 0.7918)")

hc_models = {
    'XGBoost':          xgb_hc_scores,
    'Neural Network':   nn_hc_scores,
    'Logistic Reg.':    lr_hc_scores,
    'Isolation Forest': iso_hc_scores,
}

# ──────────────────────────────────────────────────────────────
# Oracle data (sklearn equivalents — AUC may differ slightly)
# ──────────────────────────────────────────────────────────────

print("\n[2/2] Loading Oracle data...")
or_df = pd.read_csv(os.path.join(BASE_DIR, 'dataset', 'fraud_oracle_preprocessed.csv'))
y_or  = or_df['FraudFound_P'].astype(int)
X_or  = pd.get_dummies(or_df.drop(columns=['FraudFound_P']), drop_first=True)

X_or_train, X_or_test, y_or_train, y_or_test = train_test_split(
    X_or, y_or, test_size=0.2, random_state=42, stratify=y_or
)
X_or_tr2, X_or_val2, y_or_tr2, y_or_val2 = train_test_split(
    X_or_train, y_or_train, test_size=0.1, random_state=42, stratify=y_or_train
)

scaler_or  = StandardScaler()
X_or_tr_s  = scaler_or.fit_transform(X_or_train)
X_or_test_s = scaler_or.transform(X_or_test)
X_or_tr2_s  = scaler_or.fit_transform(X_or_tr2)
X_or_val2_s = scaler_or.transform(X_or_val2)
X_or_test2_s = scaler_or.transform(X_or_test)

# Logistic Regression (sklearn — class-balanced)
print("  Training Logistic Regression (Oracle)...")
lr_or = SklearnLR(class_weight='balanced', C=0.1, max_iter=1000, random_state=42, solver='lbfgs')
lr_or.fit(X_or_tr_s, y_or_train)
lr_or_scores = lr_or.predict_proba(X_or_test_s)[:, 1]
print(f"    AUC = {roc_auc_score(y_or_test, lr_or_scores):.4f}  (reported: 0.783)")

# XGBoost
print("  Training XGBoost (Oracle)...")
dist_or = y_or_tr2.value_counts().sort_index()
pos_w_or = dist_or[0] / dist_or[1]
xgb_or = XGBClassifier(
    objective='binary:logistic', eval_metric='auc',
    max_depth=6, learning_rate=0.05, subsample=0.9,
    n_estimators=500, early_stopping_rounds=30,
    scale_pos_weight=pos_w_or,
    random_state=42, tree_method='hist', verbosity=0,
)
xgb_or.fit(X_or_tr2_s, y_or_tr2, eval_set=[(X_or_val2_s, y_or_val2)], verbose=False)
xgb_or_scores = xgb_or.predict_proba(X_or_test2_s)[:, 1]
print(f"    AUC = {roc_auc_score(y_or_test, xgb_or_scores):.4f}  (reported: 0.848)")

# Isolation Forest
print("  Training Isolation Forest (Oracle)...")
iso_or = IsolationForest(n_estimators=200, contamination=0.06, random_state=42, n_jobs=-1)
iso_or.fit(X_or_tr_s)
iso_or_scores = -iso_or.score_samples(X_or_test_s)
print(f"    AUC = {roc_auc_score(y_or_test, iso_or_scores):.4f}  (reported: 0.507)")

or_models = {
    'XGBoost':          xgb_or_scores,
    'Logistic Reg.':    lr_or_scores,
    'Isolation Forest': iso_or_scores,
}

# ══════════════════════════════════════════════════════════════
# FIGURE 1 — ROC Curves (Healthcare + Oracle side-by-side)
# ══════════════════════════════════════════════════════════════

print("\nGenerating figures...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('ROC Curves — All Models', fontsize=14, fontweight='bold')

panels = [
    ('Healthcare  —  Medicare Provider Billing\n9.4% fraud rate  |  5,410 providers',
     hc_models, y_hc_test),
    ('Oracle  —  Vehicle Insurance Claims\n6% fraud rate  |  15,420 claims',
     or_models, y_or_test),
]

for ax, (title, models, y_true) in zip(axes, panels):
    for name, scores in models.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = auc(fpr, tpr)
        ls = '--' if name == 'Isolation Forest' else '-'
        ax.plot(fpr, tpr, color=COLORS[name], linestyle=ls, linewidth=2.0,
                label=f'{name}  (AUC = {auc_val:.3f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle=':', linewidth=1.2, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim([0, 1]);  ax.set_ylim([0, 1.02])

plt.tight_layout()
save('roc_curves.png')

# ══════════════════════════════════════════════════════════════
# FIGURE 2 — Precision-Recall Curves (Healthcare)
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.set_title(
    'Precision-Recall Curves  —  Healthcare Fraud\n'
    'Medicare provider-level billing  |  9.4% fraud rate',
    fontsize=12, fontweight='bold'
)

baseline = y_hc_test.mean()
ax.axhline(y=baseline, color='grey', linestyle=':', linewidth=1.2,
           label=f'Random baseline  ({baseline:.3f})')

for name, scores in hc_models.items():
    prec, rec, _ = precision_recall_curve(y_hc_test, scores)
    ap = average_precision_score(y_hc_test, scores)
    ls = '--' if name == 'Isolation Forest' else '-'
    ax.plot(rec, prec, color=COLORS[name], linestyle=ls, linewidth=2.0,
            label=f'{name}  (AP = {ap:.3f})')

ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.legend(fontsize=9.5, loc='upper right')
ax.set_xlim([0, 1]);  ax.set_ylim([0, 1.02])
plt.tight_layout()
save('pr_curves.png')

# ══════════════════════════════════════════════════════════════
# FIGURE 3 — XGBoost Feature Importance (Healthcare, top 20)
# ══════════════════════════════════════════════════════════════

importances   = xgb_hc.feature_importances_
feat_names    = X_hc.columns.tolist()
total_imp     = importances.sum()

top20 = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:20]
names_plot = [n.replace('_', ' ') for n, _ in reversed(top20)]
vals_plot  = [v for _, v in reversed(top20)]
colors_plot = ['#C0392B' if n == 'total reimbursed' else COLORS['XGBoost'] for n in names_plot]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(names_plot)), vals_plot, color=colors_plot, alpha=0.85, height=0.7)

ax.set_yticks(range(len(names_plot)))
ax.set_yticklabels(names_plot, fontsize=9.5)
ax.set_xlabel('Feature Importance (gain)', fontsize=11)
ax.set_title(
    'XGBoost Feature Importance  —  Healthcare Fraud (Top 20)\n'
    'Medicare provider-level aggregation  |  red bar = total_reimbursed',
    fontsize=12, fontweight='bold'
)

for bar, (name, val) in zip(bars, reversed(top20)):
    pct = val / total_imp * 100
    if pct >= 1.5:
        ax.text(val + total_imp * 0.003, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%', va='center', fontsize=8.5)

plt.tight_layout()
save('feature_importance.png')

# ══════════════════════════════════════════════════════════════
# FIGURE 4 — Threshold Sensitivity (XGBoost Healthcare)
# ══════════════════════════════════════════════════════════════

thresh_range = np.linspace(0.01, 0.99, 300)
f2s, recs, precs = [], [], []

for t in thresh_range:
    y_pred = (xgb_hc_scores >= t).astype(int)
    f2s.append(fbeta_score(y_hc_test, y_pred, beta=2, zero_division=0))
    recs.append(recall_score(y_hc_test, y_pred, zero_division=0))
    precs.append(precision_score(y_hc_test, y_pred, zero_division=0))

best_t_idx = int(np.argmax(f2s))
best_t     = thresh_range[best_t_idx]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresh_range, recs,  color='#C0392B', linewidth=2.0, label='Recall')
ax.plot(thresh_range, precs, color='#2980B9', linewidth=2.0, label='Precision')
ax.plot(thresh_range, f2s,   color='#27AE60', linewidth=2.5, label='F2 Score')

ax.axvline(x=best_t, color='#555555', linestyle='--', linewidth=1.8,
           label=f'Optimal threshold ({best_t:.2f})  →  F2 = {f2s[best_t_idx]:.3f}')
ax.axvline(x=0.50,   color='black',   linestyle=':',  linewidth=1.2, alpha=0.5,
           label='Default threshold (0.50)')

ax.fill_betweenx([0, 1], 0, best_t, alpha=0.04, color='#27AE60')

ax.set_xlabel('Decision Threshold', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title(
    'Threshold Sensitivity  —  XGBoost Healthcare\n'
    'Trade-off between Recall, Precision, and F2 as the decision threshold varies',
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=10)
ax.set_xlim([0, 1]);  ax.set_ylim([0, 1.05])
plt.tight_layout()
save('threshold_sensitivity.png')

print("\nDone. All figures saved to figures/")
