"""
FreshBasket Customer Churn Prediction
Phase 3: Customer-Level Batch Churn Prediction & Local SHAP Attribution Engine

Generates standardized customer-level predictions for all eligible customers:
- Preserves the existing saved Random Forest model pipeline
- Does not modify existing model weights or retraining
- Predicts churn probability for all 2,549 eligible pre-cutoff customers (and holds out 51 post-cutoff signups)
- Uses default decision threshold tau = 0.50
- Computes customer-level SHAP contributions using TreeExplainer for the churn class (Class 1)
- Identifies the top two drivers by absolute contribution magnitude with preserved signs
- Exports:
    1. phases/phase3/output/phase3_batch_predictions_all_eligible.csv (All 2,549 eligible customers)
    2. phases/phase3/output/phase3_validation_predictions_with_drivers.csv (602 validation cohort members)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

# Safe path resolution
try:
    from phases.paths import (
        PROJECT_ROOT,
        SAVED_MODEL_PATH,
        PROCESSED_DATASET_PATH,
        PHASE3_OUTPUT
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SAVED_MODEL_PATH = PROJECT_ROOT / "models" / "phase3_random_forest_pipeline.joblib"
    PROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv"
    PHASE3_OUTPUT = PROJECT_ROOT / "phases" / "phase3" / "output"

PHASE3_OUTPUT.mkdir(parents=True, exist_ok=True)

# Exact 28 feature schema expected by the pipeline
FEATURE_COLUMNS = [
    'AGE', 'AGE_IS_MISSING', 'GENDER', 'CITY', 'MEMBERSHIP_TIER',
    'HOME_STORE_PRICE_TIER', 'ELIGIBLE_PRE_CUTOFF_MONTHS', 'TENURE_DAYS_PRE_CUTOFF',
    'TOTAL_TXNS_PRE_CUTOFF', 'TOTAL_SPEND_PRE_CUTOFF', 'ACTIVE_MONTHS_PRE_CUTOFF',
    'PRE_CUTOFF_OBSERVED_MONTHS', 'TOTAL_APP_SESSIONS_PRE_CUTOFF',
    'TOTAL_EMAILS_OPENED_PRE_CUTOFF', 'TOTAL_COUPONS_REDEEMED_PRE_CUTOFF',
    'TOTAL_SUPPORT_TICKETS_PRE_CUTOFF', 'COMPLAINT_RECORDED_PRE_CUTOFF',
    'TXNS_LAST_3M', 'SPEND_LAST_3M', 'RECENCY_MONTHS_PRE_CUTOFF',
    'HAS_NO_PRE_CUTOFF_ACTIVITY', 'AVG_MONTHLY_TXNS_PRE_CUTOFF',
    'AVG_MONTHLY_SPEND_PRE_CUTOFF', 'AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF',
    'AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF', 'AVG_BASKET_VALUE_PRE_CUTOFF',
    'NEVER_PURCHASED_PRE_CUTOFF', 'IS_LIMITED_PRE_CUTOFF_HISTORY'
]

CAT_COLUMNS = ['GENDER', 'CITY', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER']
NUM_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CAT_COLUMNS]


def run_batch_prediction():
    print("=== Running Batch Customer Churn Prediction & Local SHAP Attribution ===")
    
    # 1. Load trained pipeline
    if not SAVED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model pipeline not found at '{SAVED_MODEL_PATH}'")
    pipeline = joblib.load(SAVED_MODEL_PATH)
    print(f"Loaded existing model pipeline from '{SAVED_MODEL_PATH}'")

    # 2. Ingest master dataset
    if not PROCESSED_DATASET_PATH.exists():
        raise FileNotFoundError(f"Modeling dataset not found at '{PROCESSED_DATASET_PATH}'")
    master_df = pd.read_csv(PROCESSED_DATASET_PATH)
    total_customers = len(master_df)
    print(f"Loaded master dataset: {total_customers} rows, {master_df.shape[1]} columns.")

    # 3. Filter eligible population (Cutoff: March 31, 2024)
    # Exclude 51 customers who signed up post-cutoff (April-May 2024)
    is_post_cutoff = master_df['VALIDATION_COHORT'] == 'EXCLUDED_POST_CUTOFF'
    excluded_count = is_post_cutoff.sum()
    eligible_df = master_df[~is_post_cutoff].copy().reset_index(drop=True)
    eligible_count = len(eligible_df)
    
    print(f"Population Audit:")
    print(f"  - Total source customers: {total_customers}")
    print(f"  - Excluded post-cutoff signups (April-May 2024): {excluded_count}")
    print(f"  - Eligible pre-cutoff customers for batch prediction: {eligible_count}")
    assert eligible_count == 2549, f"Expected 2,549 eligible customers, got {eligible_count}"

    # 4. Extract features
    X_eligible = eligible_df[FEATURE_COLUMNS].copy()

    # 5. Model Inference (Probabilities & Binary Predictions)
    probs = pipeline.predict_proba(X_eligible)[:, 1]
    preds = (probs >= 0.50).astype(int)

    # 6. Extract Preprocessor and RandomForest Estimator for SHAP
    preprocessor = pipeline.named_steps['pre']
    rf_clf = pipeline.named_steps['clf']

    # Transform features
    X_trans = preprocessor.transform(X_eligible)
    ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
    transformed_feature_names = NUM_COLUMNS + ohe.get_feature_names_out(CAT_COLUMNS).tolist()

    # Compute customer-level SHAP contributions
    print(f"Computing exact TreeExplainer SHAP contributions for {eligible_count} customers...")
    explainer = shap.TreeExplainer(rf_clf)
    shap_values = explainer.shap_values(X_trans)

    if isinstance(shap_values, list):
        sv_churn = shap_values[1]  # Churn class (Class 1)
        base_val = explainer.expected_value[1]
    elif len(shap_values.shape) == 3:
        sv_churn = shap_values[:, :, 1]
        base_val = explainer.expected_value[1]
    else:
        sv_churn = shap_values
        base_val = explainer.expected_value

    print(f"SHAP Explainer Base Value (Class 1 Churn): {base_val:.4f}")

    # Extract Top 2 Drivers per customer
    top_driver_1_names = []
    top_driver_1_contribs = []
    top_driver_2_names = []
    top_driver_2_contribs = []

    for i in range(eligible_count):
        row_sv = sv_churn[i]
        top_idx = np.argsort(np.abs(row_sv))[::-1][:2]
        
        idx1, idx2 = top_idx[0], top_idx[1]
        top_driver_1_names.append(transformed_feature_names[idx1])
        top_driver_1_contribs.append(round(float(row_sv[idx1]), 6))
        top_driver_2_names.append(transformed_feature_names[idx2])
        top_driver_2_contribs.append(round(float(row_sv[idx2]), 6))

    # 7. Construct Comprehensive Prediction File
    results_df = pd.DataFrame({
        'CUSTOMER_ID': eligible_df['CUSTOMER_ID'].astype(int),
        'CHURN_PROBABILITY': np.round(probs, 4),
        'PREDICTED_LABEL': preds,
        'TOP_DRIVER_1': top_driver_1_names,
        'TOP_DRIVER_1_CONTRIBUTION': top_driver_1_contribs,
        'TOP_DRIVER_2': top_driver_2_names,
        'TOP_DRIVER_2_CONTRIBUTION': top_driver_2_contribs,
        # Separate evaluation columns clearly distinguished
        'ACTUAL_CHURNED': eligible_df['CHURNED'].astype(int),
        'VALIDATION_COHORT': eligible_df['VALIDATION_COHORT'],
        'SIGNUP_DATE': eligible_df['SIGNUP_DATE'],
        'MEMBERSHIP_TIER': eligible_df['MEMBERSHIP_TIER'],
        'CITY': eligible_df['CITY']
    })

    # Validate output integrity
    assert len(results_df) == 2549, f"Expected 2549 rows, got {len(results_df)}"
    assert results_df['CUSTOMER_ID'].nunique() == 2549, "Duplicate customer IDs found!"
    assert (results_df['CHURN_PROBABILITY'] >= 0.0).all() and (results_df['CHURN_PROBABILITY'] <= 1.0).all(), "Invalid probability range!"
    assert ((results_df['CHURN_PROBABILITY'] >= 0.50).astype(int) == results_df['PREDICTED_LABEL']).all(), "Threshold 0.50 mismatch!"
    assert results_df['TOP_DRIVER_1'].notna().all(), "Null driver detected!"
    assert results_df['TOP_DRIVER_2'].notna().all(), "Null driver detected!"

    # 8. Save CSV outputs
    batch_out_path = PHASE3_OUTPUT / "phase3_batch_predictions_all_eligible.csv"
    results_df.to_csv(batch_out_path, index=False)
    print(f"SUCCESS: Saved batch prediction file for all {len(results_df)} eligible customers to '{batch_out_path}'.")

    # Validation cohort subset with drivers
    val_subset = results_df[results_df['VALIDATION_COHORT'] == 'VAL'].copy().reset_index(drop=True)
    val_out_path = PHASE3_OUTPUT / "phase3_validation_predictions_with_drivers.csv"
    val_subset.to_csv(val_out_path, index=False)
    print(f"SUCCESS: Saved validation cohort predictions with drivers ({len(val_subset)} rows) to '{val_out_path}'.")

    # Summary statistics
    overall_predicted_churners = results_df['PREDICTED_LABEL'].sum()
    val_predicted_churners = val_subset['PREDICTED_LABEL'].sum()
    print(f"\nPrediction Summary:")
    print(f"  - All Eligible ({len(results_df)}): Predicted At-Risk = {overall_predicted_churners} ({overall_predicted_churners/len(results_df)*100:.2f}%), Actual Churners = {results_df['ACTUAL_CHURNED'].sum()}")
    print(f"  - Validation Cohort ({len(val_subset)}): Predicted At-Risk = {val_predicted_churners} ({val_predicted_churners/len(val_subset)*100:.2f}%), Actual Churners = {val_subset['ACTUAL_CHURNED'].sum()}")
    
    # Display sample rows
    print("\nSample Output (First 3 Customers):")
    print(results_df[['CUSTOMER_ID', 'CHURN_PROBABILITY', 'PREDICTED_LABEL', 'TOP_DRIVER_1', 'TOP_DRIVER_1_CONTRIBUTION', 'TOP_DRIVER_2', 'TOP_DRIVER_2_CONTRIBUTION']].head(3).to_string(index=False))

    return results_df


if __name__ == '__main__':
    run_batch_prediction()
