"""
FreshBasket Customer Churn Prediction
Phase 4 — Step 2: Feature Ablation Study
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)

# Safe path resolution
try:
    from phases.paths import (
        PROJECT_ROOT,
        PROCESSED_DATASET_PATH,
        PHASE4_OUTPUT as OUTPUTS_ANALYSIS,
        PHASE4_CHARTS as CHARTS_DIR
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv"
    OUTPUTS_ANALYSIS = PROJECT_ROOT / "phases" / "phase4" / "output"
    CHARTS_DIR = OUTPUTS_ANALYSIS / "charts"

OUTPUTS_ANALYSIS.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)




def _find_dataset():
    if PROCESSED_DATASET_PATH.exists():
        return PROCESSED_DATASET_PATH
    if (PROJECT_ROOT / "phase2_modeling_dataset.csv").exists():
        return PROJECT_ROOT / "phase2_modeling_dataset.csv"
    return PROCESSED_DATASET_PATH


def run_ablation_study():
    dataset_path = _find_dataset()
    df_master = pd.read_csv(dataset_path)

    train_df = df_master[df_master['VALIDATION_COHORT'] == 'TRAIN'].copy()
    val_df = df_master[df_master['VALIDATION_COHORT'] == 'VAL'].copy()

    # Define the 4 Feature Sets
    set1_cols = [
        'AGE', 'AGE_IS_MISSING', 'GENDER', 'CITY', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER'
    ]
    set2_cols = set1_cols + [
        'ELIGIBLE_PRE_CUTOFF_MONTHS', 'TENURE_DAYS_PRE_CUTOFF', 'PRE_CUTOFF_OBSERVED_MONTHS', 'IS_LIMITED_PRE_CUTOFF_HISTORY'
    ]
    set3_cols = set2_cols + [
        'TOTAL_TXNS_PRE_CUTOFF', 'TOTAL_SPEND_PRE_CUTOFF', 'ACTIVE_MONTHS_PRE_CUTOFF',
        'RECENCY_MONTHS_PRE_CUTOFF', 'TXNS_LAST_3M', 'SPEND_LAST_3M',
        'HAS_NO_PRE_CUTOFF_ACTIVITY', 'AVG_MONTHLY_TXNS_PRE_CUTOFF', 'AVG_MONTHLY_SPEND_PRE_CUTOFF',
        'AVG_BASKET_VALUE_PRE_CUTOFF', 'NEVER_PURCHASED_PRE_CUTOFF', 'TOTAL_COUPONS_REDEEMED_PRE_CUTOFF'
    ]
    admin_cols = ['CUSTOMER_ID', 'SIGNUP_DATE', 'CHURNED', 'VALIDATION_COHORT', 'IS_POST_CUTOFF_SIGNUP']
    set4_cols = [c for c in df_master.columns if c not in admin_cols]

    feature_sets = [
        ('Set 1: Demographics and Membership', set1_cols),
        ('Set 2: Demographics plus History', set2_cols),
        ('Set 3: Purchase Behavior', set3_cols),
        ('Set 4: Full Feature Set (Reference)', set4_cols)
    ]

    all_cat_cols = ['GENDER', 'CITY', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER']
    y_train = train_df['CHURNED']
    y_val = val_df['CHURNED']

    results_list = []

    for name, cols in feature_sets:
        curr_cat_cols = [c for c in cols if c in all_cat_cols]
        curr_num_cols = [c for c in cols if c not in all_cat_cols]
        
        num_tf = Pipeline([('imputer', SimpleImputer(strategy='median'))])
        cat_tf = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', num_tf, curr_num_cols),
            ('cat', cat_tf, curr_cat_cols)
        ])
        
        rf = Pipeline([
            ('pre', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=42, class_weight='balanced'))
        ])
        
        rf.fit(train_df[cols], y_train)
        preds = rf.predict(val_df[cols])
        proba = rf.predict_proba(val_df[cols])[:, 1]
        
        cm = confusion_matrix(y_val, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        results_list.append({
            'Feature_Set': name,
            'N_Features': len(cols),
            'Accuracy': round(accuracy_score(y_val, preds), 4),
            'Precision': round(precision_score(y_val, preds, zero_division=0), 4),
            'Recall': round(recall_score(y_val, preds, zero_division=0), 4),
            'F1_Score': round(f1_score(y_val, preds, zero_division=0), 4),
            'PR_AUC': round(average_precision_score(y_val, proba), 4),
            'ROC_AUC': round(roc_auc_score(y_val, proba), 4),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
        })

    df_results = pd.DataFrame(results_list)
    ref_row = df_results[df_results['Feature_Set'].str.contains('Full Feature Set')].iloc[0]

    delta_list = []
    for idx, row in df_results.iterrows():
        delta_list.append({
            'Feature_Set': row['Feature_Set'],
            'N_Features': row['N_Features'],
            'Accuracy': row['Accuracy'],
            'Delta_Accuracy': round(row['Accuracy'] - ref_row['Accuracy'], 4),
            'Precision': row['Precision'],
            'Delta_Precision': round(row['Precision'] - ref_row['Precision'], 4),
            'Recall': row['Recall'],
            'Delta_Recall': round(row['Recall'] - ref_row['Recall'], 4),
            'F1_Score': row['F1_Score'],
            'Delta_F1': round(row['F1_Score'] - ref_row['F1_Score'], 4),
            'PR_AUC': row['PR_AUC'],
            'Delta_PR_AUC': round(row['PR_AUC'] - ref_row['PR_AUC'], 4),
            'ROC_AUC': row['ROC_AUC'],
            'Delta_ROC_AUC': round(row['ROC_AUC'] - ref_row['ROC_AUC'], 4),
            'TP': row['TP'], 'FP': row['FP'], 'FN': row['FN'], 'TN': row['TN']
        })
    df_deltas = pd.DataFrame(delta_list)

    # Export tables
    df_results.to_csv(OUTPUTS_ANALYSIS / 'phase4_step2_ablation_results.csv', index=False)
    df_deltas.to_csv(OUTPUTS_ANALYSIS / 'phase4_step2_ablation_deltas.csv', index=False)

    # Visualization
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    short_names = ['Set 1\nDemog\n(6)', 'Set 2\nDemog+Hist\n(10)', 'Set 3\nPurchase\n(22)', 'Set 4\nFull (Ref)\n(28)']

    x = np.arange(len(short_names))
    width = 0.35
    ax1 = axes[0]
    rects1 = ax1.bar(x - width/2, df_results['PR_AUC'], width, label='PR-AUC (Minority Class)', color='#ff7f0e', alpha=0.9)
    rects2 = ax1.bar(x + width/2, df_results['ROC_AUC'], width, label='ROC-AUC (Global Ranking)', color='#1f77b4', alpha=0.85)
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Ablation Study: Curve Performance (PR-AUC vs. ROC-AUC)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.1412, color='red', linestyle='--', alpha=0.7, label='Val Base Churn Rate (0.141)')
    ax1.legend(loc='upper left', framealpha=0.9)

    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2 = axes[1]
    w3 = 0.25
    r_rec = ax2.bar(x - w3, df_results['Recall'], w3, label='Recall (Churners Caught)', color='#2ca02c', alpha=0.85)
    r_prec = ax2.bar(x, df_results['Precision'], w3, label='Precision (Alert Accuracy)', color='#9467bd', alpha=0.85)
    r_f1 = ax2.bar(x + w3, df_results['F1_Score'], w3, label='F1-Score (Harmonic Mean)', color='#d62728', alpha=0.85)
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Ablation Study: Classification Trade-offs (Threshold = 0.50)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='upper left', framealpha=0.9)

    for rect in r_rec:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for rect in r_prec:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for rect in r_f1:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'phase4_ablation_comparison.png', dpi=300)
    plt.close()

    print("\n=== Phase 4 Step 2 Script Execution Complete ===")


if __name__ == '__main__':
    run_ablation_study()
