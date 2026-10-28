"""
FreshBasket Customer Churn Prediction
Phase 3: Model Training Evaluation Script
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)

# Safe path resolution
try:
    from phases.paths import PROJECT_ROOT, PROCESSED_DATASET_PATH
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv"


def _find_dataset():
    if PROCESSED_DATASET_PATH.exists():
        return PROCESSED_DATASET_PATH
    if (PROJECT_ROOT / "phase2_modeling_dataset.csv").exists():
        return PROJECT_ROOT / "phase2_modeling_dataset.csv"
    return PROCESSED_DATASET_PATH


def main():
    dataset_path = _find_dataset()
    df = pd.read_csv(dataset_path)

    # Split cohorts
    train_df = df[df['VALIDATION_COHORT'] == 'TRAIN'].copy()
    val_df = df[df['VALIDATION_COHORT'] == 'VAL'].copy()

    print(f"Train cohort size: {len(train_df)}, Churned: {train_df['CHURNED'].sum()} ({train_df['CHURNED'].mean()*100:.2f}%)")
    print(f"Val cohort size: {len(val_df)}, Churned: {val_df['CHURNED'].sum()} ({val_df['CHURNED'].mean()*100:.2f}%)")

    admin_cols = ['CUSTOMER_ID', 'SIGNUP_DATE', 'CHURNED', 'VALIDATION_COHORT', 'IS_POST_CUTOFF_SIGNUP']
    feature_cols = [c for c in df.columns if c not in admin_cols]

    cat_cols = ['GENDER', 'CITY', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER']
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_train = train_df[feature_cols]
    y_train = train_df['CHURNED']

    X_val = val_df[feature_cols]
    y_val = val_df['CHURNED']

    # Preprocessor for Logistic Regression (with scaling)
    num_transformer_lr = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessor for Random Forest (without scaling)
    num_transformer_rf = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor_lr = ColumnTransformer([
        ('num', num_transformer_lr, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    preprocessor_rf = ColumnTransformer([
        ('num', num_transformer_rf, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # 1. Logistic Regression Baseline
    lr = Pipeline([
        ('pre', preprocessor_lr),
        ('clf', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
    ])

    lr.fit(X_train, y_train)
    y_val_pred_lr = lr.predict(X_val)
    y_val_proba_lr = lr.predict_proba(X_val)[:, 1]

    # 2. Random Forest
    rf = Pipeline([
        ('pre', preprocessor_rf),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=42, class_weight='balanced'))
    ])

    rf.fit(X_train, y_train)
    y_val_pred_rf = rf.predict(X_val)
    y_val_proba_rf = rf.predict_proba(X_val)[:, 1]

    def eval_model(name, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        cm = confusion_matrix(y_true, y_pred)
        print(f"=== {name} ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"PR-AUC:    {pr_auc:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
        print()
        return {
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'PR-AUC': pr_auc,
            'ROC-AUC': roc_auc,
            'TN': cm[0,0], 'FP': cm[0,1],
            'FN': cm[1,0], 'TP': cm[1,1]
        }

    res_lr = eval_model("Logistic Regression (Baseline)", y_val, y_val_pred_lr, y_val_proba_lr)
    res_rf = eval_model("Random Forest Classifier", y_val, y_val_pred_rf, y_val_proba_rf)


if __name__ == '__main__':
    main()
