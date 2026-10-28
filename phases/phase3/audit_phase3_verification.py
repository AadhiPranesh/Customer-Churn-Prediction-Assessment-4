import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)

# Robust path resolution
try:
    from phases.paths import PROJECT_ROOT, VALIDATION_PREDS_PATH
    csv_path = VALIDATION_PREDS_PATH
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    csv_path = PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions.csv"
if not csv_path.exists():
    csv_path = PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions.csv"
df = pd.read_csv(csv_path)
print("=== 1. RECALCULATION FROM PREDICTIONS CSV (N=602) ===")
models = [
    ('Logistic Regression (Baseline)', 'LR_CHURN_PROBABILITY', 'LR_PRED_LABEL'),
    ('Random Forest Classifier (Primary)', 'RF_CHURN_PROBABILITY', 'RF_PRED_LABEL')
]

for mod, prob_col, pred_col in models:
    y_true = df['CHURNED']
    y_pred = df[pred_col]
    y_prob = df[prob_col]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"Model: {mod}")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"  PR-AUC:    {average_precision_score(y_true, y_prob):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

print("\n=== 2. SEGMENT RECONCILIATION FOR ALL 602 CUSTOMERS ===")
print("MEMBERSHIP_TIER Breakdown:")
for tier in ['Silver', 'Gold', 'Platinum']:
    sub = df[df['MEMBERSHIP_TIER'] == tier]
    ch = int(sub['CHURNED'].sum())
    n = len(sub)
    print(f"  {tier:10}: N={n:3}, Churned={ch:2} ({ch/n*100:4.1f}%)")
print(f"  Total Sum : N={len(df):3}, Churned={int(df['CHURNED'].sum()):2}")

print("\nHOME_STORE_PRICE_TIER Breakdown (including Unknown):")
for pt in ['Mainstream', 'Value', 'Upscale', 'Unknown']:
    sub = df[df['HOME_STORE_PRICE_TIER'] == pt]
    ch = int(sub['CHURNED'].sum())
    n = len(sub)
    print(f"  {pt:10}: N={n:3}, Churned={ch:2} ({ch/n*100:4.1f}%)")
sum_n = sum(len(df[df['HOME_STORE_PRICE_TIER'] == pt]) for pt in ['Mainstream', 'Value', 'Upscale', 'Unknown'])
sum_ch = sum(int(df[df['HOME_STORE_PRICE_TIER'] == pt]['CHURNED'].sum()) for pt in ['Mainstream', 'Value', 'Upscale', 'Unknown'])
print(f"  Total Sum : N={sum_n:3}, Churned={sum_ch:2}")

print("\n=== 3. VERIFY METRICS BY SEGMENT ===")
for seg_col in ['MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER']:
    print(f"\n--- {seg_col} Detailed Metrics ---")
    cats = sorted(df[seg_col].unique())
    for cat in cats:
        sub = df[df[seg_col] == cat]
        y_true = sub['CHURNED']
        n = len(sub)
        ch = int(y_true.sum())
        for m_name, pred_col in [('LR', 'LR_PRED_LABEL'), ('RF', 'RF_PRED_LABEL')]:
            y_pred = sub[pred_col]
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            print(f"  {cat:10} | {m_name:2} | N={n:3} | Churn={ch:2} | Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f} | TP={tp:2} FP={fp:2} FN={fn:2} TN={tn:3}")
