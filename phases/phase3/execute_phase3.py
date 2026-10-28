"""
FreshBasket Customer Churn Prediction
Phase 3: Execution and Evaluation Script
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
    from phases.paths import (
        PROJECT_ROOT,
        PROCESSED_DATASET_PATH,
        PHASE3_OUTPUT,
        VALIDATION_PREDS_PATH,
        MODEL_METRICS_PATH,
        LR_COEFS_PATH,
        RF_IMPORTANCES_PATH
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv"
    PHASE3_OUTPUT = PROJECT_ROOT / "phases" / "phase3" / "output"
    VALIDATION_PREDS_PATH = PHASE3_OUTPUT / "phase3_validation_predictions.csv"
    MODEL_METRICS_PATH = PHASE3_OUTPUT / "phase3_model_metrics.csv"
    LR_COEFS_PATH = PHASE3_OUTPUT / "phase3_logistic_regression_coefficients.csv"
    RF_IMPORTANCES_PATH = PHASE3_OUTPUT / "phase3_random_forest_feature_importances.csv"

PHASE3_OUTPUT.mkdir(parents=True, exist_ok=True)




def _find_dataset():
    if PROCESSED_DATASET_PATH.exists():
        return PROCESSED_DATASET_PATH
    if (PROJECT_ROOT / "phase2_modeling_dataset.csv").exists():
        return PROJECT_ROOT / "phase2_modeling_dataset.csv"
    return PROCESSED_DATASET_PATH


def execute_phase3():
    # 1. Load Phase 2 modeling dataset
    dataset_path = _find_dataset()
    df = pd.read_csv(dataset_path)

    # Verify cohort splits
    train_df = df[df['VALIDATION_COHORT'] == 'TRAIN'].copy()
    val_df = df[df['VALIDATION_COHORT'] == 'VAL'].copy()
    excluded_df = df[df['VALIDATION_COHORT'] == 'EXCLUDED_POST_CUTOFF'].copy()

    print(f"Training Cohort: {len(train_df)} rows, Churned = {train_df['CHURNED'].sum()} ({train_df['CHURNED'].mean()*100:.2f}%)")
    print(f"Validation Cohort: {len(val_df)} rows, Churned = {val_df['CHURNED'].sum()} ({val_df['CHURNED'].mean()*100:.2f}%)")
    print(f"Excluded Post-Cutoff: {len(excluded_df)} rows")

    admin_cols = ['CUSTOMER_ID', 'SIGNUP_DATE', 'CHURNED', 'VALIDATION_COHORT', 'IS_POST_CUTOFF_SIGNUP']
    feature_cols = [c for c in df.columns if c not in admin_cols]

    cat_cols = ['GENDER', 'CITY', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER']
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_train = train_df[feature_cols]
    y_train = train_df['CHURNED']

    X_val = val_df[feature_cols]
    y_val = val_df['CHURNED']

    # Preprocessing Pipelines (Fit on Train only)
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

    preprocessor_lr = ColumnTransformer([
        ('num', num_transformer_lr, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    preprocessor_rf = ColumnTransformer([
        ('num', num_transformer_rf, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # 2. Train Models
    # Baseline: Logistic Regression
    lr_model = Pipeline([
        ('pre', preprocessor_lr),
        ('clf', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
    ])
    lr_model.fit(X_train, y_train)

    # Machine Learning: Random Forest Classifier
    rf_model = Pipeline([
        ('pre', preprocessor_rf),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=42, class_weight='balanced'))
    ])
    rf_model.fit(X_train, y_train)

    # 3. Predict on Validation Cohort
    val_preds = val_df[['CUSTOMER_ID', 'SIGNUP_DATE', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER', 'CHURNED']].copy()
    val_preds['LR_CHURN_PROBABILITY'] = lr_model.predict_proba(X_val)[:, 1].round(4)
    val_preds['LR_PRED_LABEL'] = lr_model.predict(X_val)

    val_preds['RF_CHURN_PROBABILITY'] = rf_model.predict_proba(X_val)[:, 1].round(4)
    val_preds['RF_PRED_LABEL'] = rf_model.predict(X_val)

    val_preds['PRIMARY_MODEL_NAME'] = 'RandomForest_Balanced'
    val_preds['PRIMARY_MODEL_PROBABILITY'] = val_preds['RF_CHURN_PROBABILITY']
    val_preds['PRIMARY_MODEL_PRED_LABEL'] = val_preds['RF_PRED_LABEL']

    # Save to canonical paths
    val_preds.to_csv(VALIDATION_PREDS_PATH, index=False)
    print(f"Saved predictions to '{VALIDATION_PREDS_PATH}'.")

    # 4. Metrics Table
    def get_metrics(name, y_true, y_pred, y_proba):
        cm = confusion_matrix(y_true, y_pred)
        return {
            'Model': name,
            'Accuracy': round(accuracy_score(y_true, y_pred), 4),
            'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
            'F1_Score': round(f1_score(y_true, y_pred, zero_division=0), 4),
            'PR_AUC': round(average_precision_score(y_true, y_proba), 4),
            'ROC_AUC': round(roc_auc_score(y_true, y_proba), 4),
            'TN': cm[0,0],
            'FP': cm[0,1],
            'FN': cm[1,0],
            'TP': cm[1,1]
        }

    metrics_list = [
        get_metrics('Logistic Regression (Baseline)', y_val, val_preds['LR_PRED_LABEL'], val_preds['LR_CHURN_PROBABILITY']),
        get_metrics('Random Forest Classifier (Primary)', y_val, val_preds['RF_PRED_LABEL'], val_preds['RF_CHURN_PROBABILITY'])
    ]
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(MODEL_METRICS_PATH, index=False)
    print("\n=== Model Metrics on Validation Cohort (N=602) ===")
    print(metrics_df.to_string(index=False))

    # 5. Extract Feature Importances and Coefficients
    ohe = lr_model.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names

    # LR Coefficients
    lr_coefs = lr_model.named_steps['clf'].coef_[0]
    df_lr_imp = pd.DataFrame({
        'Feature': all_feature_names,
        'LR_Coefficient': lr_coefs,
        'Odds_Ratio': np.exp(lr_coefs)
    }).sort_values(by='LR_Coefficient', ascending=False)
    df_lr_imp.to_csv(LR_COEFS_PATH, index=False)

    # RF Feature Importances
    rf_importances = rf_model.named_steps['clf'].feature_importances_
    df_rf_imp = pd.DataFrame({
        'Feature': all_feature_names,
        'RF_Gini_Importance': rf_importances
    }).sort_values(by='RF_Gini_Importance', ascending=False)
    df_rf_imp.to_csv(RF_IMPORTANCES_PATH, index=False)

    print("\nTop 5 Churn Drivers (Random Forest):")
    print(df_rf_imp.head(5).to_string(index=False))

    print("\nTop 5 Protective Signals (Logistic Regression):")
    print(df_lr_imp.tail(5).to_string(index=False))


if __name__ == '__main__':
    execute_phase3()
