"""
FreshBasket Customer Churn Prediction
Phase 3 Sensitivity Analysis: Decision Threshold Comparison
Evaluates default threshold (0.50) against training-only 5-fold CV F1-maximizing threshold.
Does NOT select any threshold using validation data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# Anchor root path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phases.paths import PROCESSED_DATASET_PATH, VALIDATION_PREDS_PATH, PHASE3_OUTPUT


def run_threshold_analysis():
    # 1. Load dataset
    df = pd.read_csv(PROCESSED_DATASET_PATH)
    train_df = df[df['VALIDATION_COHORT'] == 'TRAIN'].copy()
    val_df = df[df['VALIDATION_COHORT'] == 'VAL'].copy()

    admin_cols = ['CUSTOMER_ID', 'SIGNUP_DATE', 'CHURNED', 'VALIDATION_COHORT', 'IS_POST_CUTOFF_SIGNUP']
    feature_cols = [c for c in df.columns if c not in admin_cols]
    cat_cols = ['GENDER', 'CITY', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER']
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_train = train_df[feature_cols]
    y_train = train_df['CHURNED'].values
    y_val = val_df['CHURNED'].values

    # Preprocessors
    num_transformer_lr = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    num_transformer_rf = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    pre_lr = ColumnTransformer([('num', num_transformer_lr, num_cols), ('cat', cat_transformer, cat_cols)])
    pre_rf = ColumnTransformer([('num', num_transformer_rf, num_cols), ('cat', cat_transformer, cat_cols)])

    lr = Pipeline([('pre', pre_lr), ('clf', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))])
    rf = Pipeline([('pre', pre_rf), ('clf', RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=42, class_weight='balanced'))])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running 5-Fold Cross-Validation on TRAINING cohort only...")
    oof_lr = cross_val_predict(lr, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    oof_rf = cross_val_predict(rf, X_train, y_train, cv=cv, method='predict_proba')[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)

    def find_best_thresh(oof_probs, y_true):
        best_t, best_f1 = 0.50, 0.0
        for t in thresholds:
            pred = (oof_probs >= t).astype(int)
            score = f1_score(y_true, pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
        return round(float(best_t), 2), best_f1

    best_t_lr, f1_lr_cv = find_best_thresh(oof_lr, y_train)
    best_t_rf, f1_rf_cv = find_best_thresh(oof_rf, y_train)

    print(f"Logistic Regression: Optimal Training-CV Threshold = {best_t_lr:.2f} (Training CV F1 = {f1_lr_cv:.4f})")
    print(f"Random Forest:       Optimal Training-CV Threshold = {best_t_rf:.2f} (Training CV F1 = {f1_rf_cv:.4f})")

    # Read existing validation predictions
    val_preds_df = pd.read_csv(VALIDATION_PREDS_PATH)
    val_prob_lr = val_preds_df['LR_CHURN_PROBABILITY'].values
    val_prob_rf = val_preds_df['RF_CHURN_PROBABILITY'].values

    records = []
    models = [
        ('Logistic Regression', val_prob_lr, best_t_lr),
        ('Random Forest', val_prob_rf, best_t_rf)
    ]

    for model_name, val_probs, cv_t in models:
        # (i) Default threshold 0.50
        pred_def = (val_probs >= 0.50).astype(int)
        acc_def = accuracy_score(y_val, pred_def)
        prec_def = precision_score(y_val, pred_def, zero_division=0)
        rec_def = recall_score(y_val, pred_def, zero_division=0)
        f1_def = f1_score(y_val, pred_def, zero_division=0)
        tn_def, fp_def, fn_def, tp_def = confusion_matrix(y_val, pred_def).ravel()

        records.append({
            'Model': model_name,
            'Threshold_Strategy': 'Default (Official)',
            'Threshold': 0.50,
            'Accuracy': round(acc_def, 4),
            'Precision': round(prec_def, 4),
            'Recall': round(rec_def, 4),
            'F1_Score': round(f1_def, 4),
            'TN': int(tn_def),
            'FP': int(fp_def),
            'FN': int(fn_def),
            'TP': int(tp_def)
        })

        # (ii) Training-CV optimal threshold
        pred_cv = (val_probs >= cv_t).astype(int)
        acc_cv = accuracy_score(y_val, pred_cv)
        prec_cv = precision_score(y_val, pred_cv, zero_division=0)
        rec_cv = recall_score(y_val, pred_cv, zero_division=0)
        f1_cv = f1_score(y_val, pred_cv, zero_division=0)
        tn_cv, fp_cv, fn_cv, tp_cv = confusion_matrix(y_val, pred_cv).ravel()

        records.append({
            'Model': model_name,
            'Threshold_Strategy': f'Training-CV Optimized ({cv_t:.2f})',
            'Threshold': cv_t,
            'Accuracy': round(acc_cv, 4),
            'Precision': round(prec_cv, 4),
            'Recall': round(rec_cv, 4),
            'F1_Score': round(f1_cv, 4),
            'TN': int(tn_cv),
            'FP': int(fp_cv),
            'FN': int(fn_cv),
            'TP': int(tp_cv)
        })

    out_df = pd.DataFrame(records)
    out_path = PHASE3_OUTPUT / 'phase3_threshold_comparison.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved threshold comparison to: {out_path}")
    print(out_df.to_string(index=False))
    return out_df


if __name__ == '__main__':
    run_threshold_analysis()
