"""
FreshBasket Customer Churn Prediction
Phase 2: Feature Engineering & Dataset Construction Pipeline

Constructs the leakage-safe pre-cutoff modeling dataset and exports to:
- data/processed/phase2_modeling_dataset.csv
- outputs/data_checks/phase2_modeling_dataset.csv
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Safe path resolution
try:
    from phases.paths import (
        PROJECT_ROOT,
        DATA_PROCESSED_DIR,
        DATA_INTERIM_DIR,
        PHASE2_OUTPUT,
        CLEANED_CUSTOMER_PROFILE_PATH,
        CLEANED_CHURN_LABEL_PATH,
        CLEANED_MONTHLY_ACTIVITY_PATH,
        PROCESSED_DATASET_PATH
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
    PHASE2_OUTPUT = PROJECT_ROOT / "phases" / "phase2" / "output"
    CLEANED_CUSTOMER_PROFILE_PATH = DATA_INTERIM_DIR / "cleaned_fb_customer_profile.csv"
    CLEANED_CHURN_LABEL_PATH = DATA_INTERIM_DIR / "cleaned_fb_churn_label.csv"
    CLEANED_MONTHLY_ACTIVITY_PATH = DATA_INTERIM_DIR / "cleaned_fb_monthly_activity.csv"
    PROCESSED_DATASET_PATH = DATA_PROCESSED_DIR / "phase2_modeling_dataset.csv"

DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PHASE2_OUTPUT.mkdir(parents=True, exist_ok=True)



def _find_data_file(canonical, alternatives):
    if canonical.exists():
        return canonical
    for alt in alternatives:
        if alt.exists():
            return alt
    return canonical


def build_dataset():
    cust_path = _find_data_file(
        CLEANED_CUSTOMER_PROFILE_PATH,
        [
            PROJECT_ROOT / "outputs/eda/cleaned_data/cleaned_fb_customer_profile.csv",
            PROJECT_ROOT / "data/interim/cleaned_fb_customer_profile.csv",
            PROJECT_ROOT / "archive/phase_outputs_legacy/phase_1_outputs/cleaned_data/cleaned_fb_customer_profile.csv"
        ]
    )
    churn_path = _find_data_file(
        CLEANED_CHURN_LABEL_PATH,
        [
            PROJECT_ROOT / "outputs/eda/cleaned_data/cleaned_fb_churn_label.csv",
            PROJECT_ROOT / "data/interim/cleaned_fb_churn_label.csv",
            PROJECT_ROOT / "archive/phase_outputs_legacy/phase_1_outputs/cleaned_data/cleaned_fb_churn_label.csv"
        ]
    )
    act_path = _find_data_file(
        CLEANED_MONTHLY_ACTIVITY_PATH,
        [
            PROJECT_ROOT / "outputs/eda/cleaned_data/cleaned_fb_monthly_activity.csv",
            PROJECT_ROOT / "data/interim/cleaned_fb_monthly_activity.csv",
            PROJECT_ROOT / "archive/phase_outputs_legacy/phase_1_outputs/cleaned_data/cleaned_fb_monthly_activity.csv"
        ]
    )

    # Load approved Phase 1 cleaned tables
    cust = pd.read_csv(cust_path)
    churn = pd.read_csv(churn_path)
    act = pd.read_csv(act_path)

    # Ensure correct datetime parsing
    act['MONTH'] = pd.to_datetime(act['MONTH'])
    cust['SIGNUP_DATE'] = pd.to_datetime(cust['SIGNUP_DATE'])

    cutoff_date = pd.to_datetime('2024-03-31')

    # 1. Calculate Eligible Pre-Cutoff Calendar Months
    def get_eligible_months(signup):
        if signup > cutoff_date:
            return 0
        start = max(signup, pd.to_datetime('2023-01-01'))
        return (2024 - start.year) * 12 + (3 - start.month) + 1

    cust['ELIGIBLE_PRE_CUTOFF_MONTHS'] = cust['SIGNUP_DATE'].apply(get_eligible_months)

    # Base table from Customer Profile
    df = cust[['CUSTOMER_ID', 'SIGNUP_DATE', 'AGE', 'AGE_IS_MISSING', 'GENDER_CLEANED', 'CITY', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER_CLEANED', 'ELIGIBLE_PRE_CUTOFF_MONTHS']].copy()
    df.rename(columns={
        'GENDER_CLEANED': 'GENDER',
        'HOME_STORE_PRICE_TIER_CLEANED': 'HOME_STORE_PRICE_TIER'
    }, inplace=True)

    # Merge Target CHURNED
    df = df.merge(churn[['CUSTOMER_ID', 'CHURNED']], on='CUSTOMER_ID', how='inner')

    # Calculate Tenure in days as of 2024-03-31
    df['TENURE_DAYS_PRE_CUTOFF'] = (cutoff_date - df['SIGNUP_DATE']).dt.days
    df['IS_POST_CUTOFF_SIGNUP'] = (df['SIGNUP_DATE'] > cutoff_date).astype(int)
    df['TENURE_DAYS_PRE_CUTOFF'] = df['TENURE_DAYS_PRE_CUTOFF'].clip(lower=0)

    # 2. Pre-cutoff monthly activity features (Strictly MONTH < 2024-04-01)
    act_pre = act[act['MONTH'] < '2024-04-01'].copy()

    # Recency of purchase
    act_purchases = act_pre[act_pre['TRANSACTIONS'] > 0]
    latest_p = act_purchases.groupby('CUSTOMER_ID')['MONTH'].max().reset_index()
    latest_p['RECENCY_MONTHS_PRE_CUTOFF'] = (2024 - latest_p['MONTH'].dt.year) * 12 + (3 - latest_p['MONTH'].dt.month)

    # Last 3 months (January, February, March 2024)
    act_last3m = act_pre[act_pre['MONTH'] >= '2024-01-01']
    last3m_agg = act_last3m.groupby('CUSTOMER_ID').agg(
        TXNS_LAST_3M=('TRANSACTIONS', 'sum'),
        SPEND_LAST_3M=('TOTAL_SPEND_RECONSTRUCTED', 'sum')
    ).reset_index()

    # Overall pre-cutoff aggregations
    pre_agg = act_pre.groupby('CUSTOMER_ID').agg(
        TOTAL_TXNS_PRE_CUTOFF=('TRANSACTIONS', 'sum'),
        TOTAL_SPEND_PRE_CUTOFF=('TOTAL_SPEND_RECONSTRUCTED', 'sum'),
        ACTIVE_MONTHS_PRE_CUTOFF=('TRANSACTIONS', lambda s: (s > 0).sum()),
        PRE_CUTOFF_OBSERVED_MONTHS=('MONTH', 'count'),
        TOTAL_APP_SESSIONS_PRE_CUTOFF=('APP_SESSIONS', 'sum'),
        TOTAL_EMAILS_OPENED_PRE_CUTOFF=('EMAILS_OPENED', 'sum'),
        TOTAL_COUPONS_REDEEMED_PRE_CUTOFF=('COUPONS_REDEEMED', 'sum'),
        TOTAL_SUPPORT_TICKETS_PRE_CUTOFF=('SUPPORT_TICKETS', 'sum'),
        COMPLAINT_RECORDED_PRE_CUTOFF=('COMPLAINT_FLAG', lambda s: int((s > 0).any()))
    ).reset_index()

    # Merge aggregations into master df
    df = df.merge(pre_agg, on='CUSTOMER_ID', how='left')
    df = df.merge(last3m_agg, on='CUSTOMER_ID', how='left')
    df = df.merge(latest_p[['CUSTOMER_ID', 'RECENCY_MONTHS_PRE_CUTOFF']], on='CUSTOMER_ID', how='left')

    # Indicator for no pre-cutoff activity records
    df['HAS_NO_PRE_CUTOFF_ACTIVITY'] = df['PRE_CUTOFF_OBSERVED_MONTHS'].isnull().astype(int)

    # Fill nulls for customers without pre-cutoff activity records
    df['TOTAL_TXNS_PRE_CUTOFF'] = df['TOTAL_TXNS_PRE_CUTOFF'].fillna(0).astype(int)
    df['TOTAL_SPEND_PRE_CUTOFF'] = df['TOTAL_SPEND_PRE_CUTOFF'].fillna(0.0).round(2)
    df['ACTIVE_MONTHS_PRE_CUTOFF'] = df['ACTIVE_MONTHS_PRE_CUTOFF'].fillna(0).astype(int)
    df['PRE_CUTOFF_OBSERVED_MONTHS'] = df['PRE_CUTOFF_OBSERVED_MONTHS'].fillna(0).astype(int)
    df['TOTAL_APP_SESSIONS_PRE_CUTOFF'] = df['TOTAL_APP_SESSIONS_PRE_CUTOFF'].fillna(0).astype(int)
    df['TOTAL_EMAILS_OPENED_PRE_CUTOFF'] = df['TOTAL_EMAILS_OPENED_PRE_CUTOFF'].fillna(0).astype(int)
    df['TXNS_LAST_3M'] = df['TXNS_LAST_3M'].fillna(0).astype(int)
    df['SPEND_LAST_3M'] = df['SPEND_LAST_3M'].fillna(0.0).round(2)
    df['TOTAL_COUPONS_REDEEMED_PRE_CUTOFF'] = df['TOTAL_COUPONS_REDEEMED_PRE_CUTOFF'].fillna(0).astype(int)
    df['TOTAL_SUPPORT_TICKETS_PRE_CUTOFF'] = df['TOTAL_SUPPORT_TICKETS_PRE_CUTOFF'].fillna(0).astype(int)
    df['COMPLAINT_RECORDED_PRE_CUTOFF'] = df['COMPLAINT_RECORDED_PRE_CUTOFF'].fillna(0).astype(int)

    # Normalize by ELIGIBLE_PRE_CUTOFF_MONTHS
    df['AVG_MONTHLY_TXNS_PRE_CUTOFF'] = np.where(df['ELIGIBLE_PRE_CUTOFF_MONTHS'] > 0, (df['TOTAL_TXNS_PRE_CUTOFF'] / df['ELIGIBLE_PRE_CUTOFF_MONTHS']).round(2), 0.0)
    df['AVG_MONTHLY_SPEND_PRE_CUTOFF'] = np.where(df['ELIGIBLE_PRE_CUTOFF_MONTHS'] > 0, (df['TOTAL_SPEND_PRE_CUTOFF'] / df['ELIGIBLE_PRE_CUTOFF_MONTHS']).round(2), 0.0)
    df['AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF'] = np.where(df['ELIGIBLE_PRE_CUTOFF_MONTHS'] > 0, (df['TOTAL_APP_SESSIONS_PRE_CUTOFF'] / df['ELIGIBLE_PRE_CUTOFF_MONTHS']).round(2), 0.0)
    df['AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF'] = np.where(df['ELIGIBLE_PRE_CUTOFF_MONTHS'] > 0, (df['TOTAL_EMAILS_OPENED_PRE_CUTOFF'] / df['ELIGIBLE_PRE_CUTOFF_MONTHS']).round(2), 0.0)

    # Basket value
    df['AVG_BASKET_VALUE_PRE_CUTOFF'] = np.where(df['TOTAL_TXNS_PRE_CUTOFF'] > 0, (df['TOTAL_SPEND_PRE_CUTOFF'] / df['TOTAL_TXNS_PRE_CUTOFF']).round(2), 0.0)

    # Recency and never purchased flag
    df['NEVER_PURCHASED_PRE_CUTOFF'] = df['RECENCY_MONTHS_PRE_CUTOFF'].isnull().astype(int)
    df['RECENCY_MONTHS_PRE_CUTOFF'] = df['RECENCY_MONTHS_PRE_CUTOFF'].fillna(99).astype(int)

    # Limited history pre-cutoff indicator
    df['IS_LIMITED_PRE_CUTOFF_HISTORY'] = ((df['TENURE_DAYS_PRE_CUTOFF'] < 90) | (df['PRE_CUTOFF_OBSERVED_MONTHS'] < 4)).astype(int)

    # Assign Validation Cohort
    df['VALIDATION_COHORT'] = 'EXCLUDED_POST_CUTOFF'
    df.loc[df['SIGNUP_DATE'] <= '2023-06-30', 'VALIDATION_COHORT'] = 'TRAIN'
    df.loc[(df['SIGNUP_DATE'] > '2023-06-30') & (df['SIGNUP_DATE'] <= '2024-03-31'), 'VALIDATION_COHORT'] = 'VAL'

    # Export to canonical destinations
    destinations = [
        PROCESSED_DATASET_PATH,
        PHASE2_OUTPUT / "phase2_modeling_dataset.csv"
    ]
    for dst in destinations:
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, index=False)
        print(f"Saved dataset to {dst} (Shape: {df.shape})")

    return df


if __name__ == '__main__':
    build_dataset()
