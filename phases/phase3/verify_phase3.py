import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)

try:
    from phases.paths import PROJECT_ROOT, VALIDATION_PREDS_PATH, MODEL_METRICS_PATH
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    VALIDATION_PREDS_PATH = PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions.csv"
    MODEL_METRICS_PATH = PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_model_metrics.csv"

def _find(p1, p2):
    if p1.exists(): return p1
    if p2.exists(): return p2
    return p1

val_preds_path = _find(VALIDATION_PREDS_PATH, PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions.csv")
metrics_path = _find(MODEL_METRICS_PATH, PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_model_metrics.csv")


df_val = pd.read_csv(val_preds_path)

print(f"Total validation rows: {len(df_val)}")
print("Actual churn distribution:", df_val['CHURNED'].value_counts().to_dict())

metrics_csv = pd.read_csv(metrics_path)
print("\n=== Saved Metrics CSV ===")
print(metrics_csv.to_string())

print("\n=== Recomputed from Predictions CSV ===")
models = [
    ('Logistic Regression (Baseline)', 'LR_CHURN_PROBABILITY', 'LR_PRED_LABEL'),
    ('Random Forest Classifier (Primary)', 'RF_CHURN_PROBABILITY', 'RF_PRED_LABEL')
]

for mod, prob_col, pred_col in models:
    y_true = df_val['CHURNED']
    y_pred = df_val[pred_col]
    y_prob = df_val[prob_col]
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    print(f"{mod}:")
    print(f"  Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, PR_AUC: {pr_auc:.4f}, ROC_AUC: {roc_auc:.4f}")
    print(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

print("\n=== Check Segment Tables ===")
for seg_col in ['MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER']:
    print(f"\n--- Segment: {seg_col} ---")
    for val in sorted(df_val[seg_col].dropna().unique()):
        sub = df_val[df_val[seg_col] == val]
        y_true = sub['CHURNED']
        n = len(sub)
        ch = y_true.sum()
        for mod_short, col in [('LR', 'LR_PRED_LABEL'), ('RF', 'RF_PRED_LABEL')]:
            y_pred = sub[col]
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            print(f"{val:12} | {mod_short} | N={n:3} | Churn={ch:2} ({ch/n*100:4.1f}%) | Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f} | TP={tp:2} FP={fp:2} FN={fn:2} TN={tn:3}")
