"""
FreshBasket Customer Churn Prediction
Phase 3: Bootstrap Confidence Intervals (1,000 Resamples, Seed 42)
Calculates observed, bootstrap mean, and 95% percentile intervals for
PR-AUC, ROC-AUC, Precision, Recall, and F1 (threshold = 0.50) on validation cohort (N = 602).
Also runs a paired bootstrap on the PR-AUC difference (Logistic Regression minus Random Forest).
Uses existing validation predictions; does NOT retrain models.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score
)

# Anchor root path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phases.paths import VALIDATION_PREDS_PATH, PHASE3_OUTPUT


def run_bootstrap_analysis(n_resamples: int = 1000, seed: int = 42):
    print(f"Loading validation predictions from {VALIDATION_PREDS_PATH}...")
    df = pd.read_csv(VALIDATION_PREDS_PATH)
    n = len(df)
    assert n == 602, f"Expected 602 validation customers, found {n}"

    y_true = df['CHURNED'].values
    lr_prob = df['LR_CHURN_PROBABILITY'].values
    rf_prob = df['RF_CHURN_PROBABILITY'].values

    def compute_metrics(y, prob):
        pred = (prob >= 0.50).astype(int)
        pr_auc = average_precision_score(y, prob)
        roc_auc = roc_auc_score(y, prob)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        return {
            'PR-AUC': pr_auc,
            'ROC-AUC': roc_auc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        }

    # Observed metrics on full validation cohort
    obs_lr = compute_metrics(y_true, lr_prob)
    obs_rf = compute_metrics(y_true, rf_prob)
    obs_diff = obs_lr['PR-AUC'] - obs_rf['PR-AUC']

    metric_names = ['PR-AUC', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score']
    boot_lr = {m: [] for m in metric_names}
    boot_rf = {m: [] for m in metric_names}
    boot_diff = []

    rng = np.random.default_rng(seed)

    print(f"Running {n_resamples} bootstrap resamples (seed={seed})...")
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        # Guard against zero-variance resamples
        if len(np.unique(y_b)) < 2:
            continue
        lr_p_b = lr_prob[idx]
        rf_p_b = rf_prob[idx]

        m_lr = compute_metrics(y_b, lr_p_b)
        m_rf = compute_metrics(y_b, rf_p_b)

        for m in metric_names:
            boot_lr[m].append(m_lr[m])
            boot_rf[m].append(m_rf[m])

        boot_diff.append(m_lr['PR-AUC'] - m_rf['PR-AUC'])

    rows = []
    models = [
        ('Logistic Regression', obs_lr, boot_lr),
        ('Random Forest', obs_rf, boot_rf)
    ]

    for model_name, obs, boot in models:
        for m in metric_names:
            val_obs = obs[m]
            b_mean = float(np.mean(boot[m]))
            ci_low = float(np.percentile(boot[m], 2.5))
            ci_high = float(np.percentile(boot[m], 97.5))
            rows.append({
                'Model': model_name,
                'Metric': m,
                'Observed': round(val_obs, 4),
                'Bootstrap_Mean': round(b_mean, 4),
                'CI_95_Lower': round(ci_low, 4),
                'CI_95_Upper': round(ci_high, 4),
                'Contains_Zero': ''
            })

    # Paired PR-AUC difference (LR - RF)
    diff_mean = float(np.mean(boot_diff))
    diff_low = float(np.percentile(boot_diff, 2.5))
    diff_high = float(np.percentile(boot_diff, 97.5))
    diff_contains_zero = (diff_low <= 0.0 <= diff_high)

    rows.append({
        'Model': 'Paired Difference (LR - RF)',
        'Metric': 'PR-AUC Difference',
        'Observed': round(obs_diff, 4),
        'Bootstrap_Mean': round(diff_mean, 4),
        'CI_95_Lower': round(diff_low, 4),
        'CI_95_Upper': round(diff_high, 4),
        'Contains_Zero': 'Yes' if diff_contains_zero else 'No'
    })

    out_df = pd.DataFrame(rows)
    out_path = PHASE3_OUTPUT / 'phase3_bootstrap_confidence_intervals.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved bootstrap confidence intervals to: {out_path}")
    print(out_df.to_string(index=False))

    print(f"\nPaired PR-AUC Difference 95% CI: [{diff_low:.4f}, {diff_high:.4f}]")
    print(f"Contains zero: {diff_contains_zero} (No statistically significant PR-AUC superiority between models).")
    return out_df


if __name__ == '__main__':
    run_bootstrap_analysis()
