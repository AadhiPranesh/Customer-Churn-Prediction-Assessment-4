import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell('''# FreshBasket Customer Churn Prediction
## Phase 2: Customer-Level Feature Engineering


**Project:** FreshBasket Loyalty Customer Churn Prediction (Propel Assessment)  
**Objective:** Engineer a clean, leakage-safe, reproducible customer-level modeling dataset (`CUSTOMER_ID` granularity) with target `CHURNED`.  
**Prediction Cutoff:** March 31, 2024. All features are constructed strictly from pre-cutoff activity (`MONTH < 2024-04-01`).  
**Denominator Strategy:** Monthly averages use `ELIGIBLE_PRE_CUTOFF_MONTHS` (calendar tenure months during observation window) to properly treat unrecorded dormant months as zero-activity months.  
**Scope:** Phase 2 Only (No model training, no hyperparameter tuning).'''),

    nbf.v4.new_markdown_cell('''### 1. Ingestion and Cutoff Enforcement
We load the approved Phase 1 cleaned CSV tables and establish the strict temporal boundary of **March 31, 2024**.'''),

    nbf.v4.new_code_cell('''import pandas as pd
import numpy as np

# Load approved Phase 1 cleaned tables
cust = pd.read_csv('phase_1_outputs/cleaned_data/cleaned_fb_customer_profile.csv')
churn = pd.read_csv('phase_1_outputs/cleaned_data/cleaned_fb_churn_label.csv')
act = pd.read_csv('phase_1_outputs/cleaned_data/cleaned_fb_monthly_activity.csv')

# Ensure datetime parsing
act['MONTH'] = pd.to_datetime(act['MONTH'])
cust['SIGNUP_DATE'] = pd.to_datetime(cust['SIGNUP_DATE'])

cutoff_date = pd.to_datetime('2024-03-31')

print("Loaded Cleaned Data:")
print("  Customer Profile: ", cust.shape)
print("  Churn Label:      ", churn.shape)
print("  Monthly Activity: ", act.shape)
print(f"Prediction Cutoff Date: {cutoff_date.date()}")'''),

    nbf.v4.new_markdown_cell('''### 2. Base Profile Extraction & Eligible Calendar Months Calculation
We compute `ELIGIBLE_PRE_CUTOFF_MONTHS`, representing the calendar months each member was enrolled within the January 2023–March 2024 observation window.'''),

    nbf.v4.new_code_cell('''# Calculate Eligible Pre-Cutoff Calendar Months
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

# Merge Target CHURNED (Preserved from cleaned churn table)
df = df.merge(churn[['CUSTOMER_ID', 'CHURNED']], on='CUSTOMER_ID', how='inner')

# Calculate Tenure in Days as of 2024-03-31
df['TENURE_DAYS_PRE_CUTOFF'] = (cutoff_date - df['SIGNUP_DATE']).dt.days

# Flag customers who enrolled after the cutoff
df['IS_POST_CUTOFF_SIGNUP'] = (df['SIGNUP_DATE'] > cutoff_date).astype(int)
df['TENURE_DAYS_PRE_CUTOFF'] = df['TENURE_DAYS_PRE_CUTOFF'].clip(lower=0)

print(f"Total Members: {len(df)}")
print(f"Members enrolled after cutoff (2024-03-31): {df['IS_POST_CUTOFF_SIGNUP'].sum()}")'''),

    nbf.v4.new_markdown_cell('''### 3. Pre-Cutoff Monthly Activity Aggregation (Strictly MONTH < 2024-04-01)
We filter monthly activity to records dated prior to April 1, 2024. All activity during April–June 2024 is strictly excluded to prevent target leakage.'''),

    nbf.v4.new_code_cell('''# Filter strictly to pre-cutoff activity
act_pre = act[act['MONTH'] < '2024-04-01'].copy()
print(f"Pre-cutoff monthly activity records: {len(act_pre):,} rows across {act_pre['CUSTOMER_ID'].nunique()} unique members")

# A. Recency of purchase before cutoff
act_purchases = act_pre[act_pre['TRANSACTIONS'] > 0]
latest_p = act_purchases.groupby('CUSTOMER_ID')['MONTH'].max().reset_index()
latest_p['RECENCY_MONTHS_PRE_CUTOFF'] = (2024 - latest_p['MONTH'].dt.year) * 12 + (3 - latest_p['MONTH'].dt.month)

# B. Recent momentum: Last 3 months activity (January, February, March 2024)
act_last3m = act_pre[act_pre['MONTH'] >= '2024-01-01']
last3m_agg = act_last3m.groupby('CUSTOMER_ID').agg(
    TXNS_LAST_3M=('TRANSACTIONS', 'sum'),
    SPEND_LAST_3M=('TOTAL_SPEND_RECONSTRUCTED', 'sum')
).reset_index()

# C. Overall pre-cutoff aggregates
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
).reset_index()'''),

    nbf.v4.new_markdown_cell('''### 4. Integration, Missingness Imputation, and Rate Derivation
We join all behavioral blocks into the master customer table and calculate monthly averages using `ELIGIBLE_PRE_CUTOFF_MONTHS` as the denominator.'''),

    nbf.v4.new_code_cell('''# Merge behavioral blocks
df = df.merge(pre_agg, on='CUSTOMER_ID', how='left')
df = df.merge(last3m_agg, on='CUSTOMER_ID', how='left')
df = df.merge(latest_p[['CUSTOMER_ID', 'RECENCY_MONTHS_PRE_CUTOFF']], on='CUSTOMER_ID', how='left')

# Flag customers with no pre-cutoff monthly activity records
df['HAS_NO_PRE_CUTOFF_ACTIVITY'] = df['PRE_CUTOFF_OBSERVED_MONTHS'].isnull().astype(int)

# Zero-fill for members without pre-cutoff records
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

# Use ELIGIBLE_PRE_CUTOFF_MONTHS as denominator for monthly rates
df['AVG_MONTHLY_TXNS_PRE_CUTOFF'] = np.where(df['ELIGIBLE_PRE_CUTOFF_MONTHS'] > 0, (df['TOTAL_TXNS_PRE_CUTOFF'] / df['ELIGIBLE_PRE_CUTOFF_MONTHS']).round(2), 0.0)
df['AVG_MONTHLY_SPEND_PRE_CUTOFF'] = np.where(df['ELIGIBLE_PRE_CUTOFF_MONTHS'] > 0, (df['TOTAL_SPEND_PRE_CUTOFF'] / df['ELIGIBLE_PRE_CUTOFF_MONTHS']).round(2), 0.0)
df['AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF'] = np.where(df['ELIGIBLE_PRE_CUTOFF_MONTHS'] > 0, (df['TOTAL_APP_SESSIONS_PRE_CUTOFF'] / df['ELIGIBLE_PRE_CUTOFF_MONTHS']).round(2), 0.0)
df['AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF'] = np.where(df['ELIGIBLE_PRE_CUTOFF_MONTHS'] > 0, (df['TOTAL_EMAILS_OPENED_PRE_CUTOFF'] / df['ELIGIBLE_PRE_CUTOFF_MONTHS']).round(2), 0.0)

# Basket value is per transaction
df['AVG_BASKET_VALUE_PRE_CUTOFF'] = np.where(df['TOTAL_TXNS_PRE_CUTOFF'] > 0, (df['TOTAL_SPEND_PRE_CUTOFF'] / df['TOTAL_TXNS_PRE_CUTOFF']).round(2), 0.0)

# Recency handling: flag never purchased and set sentinel 99
df['NEVER_PURCHASED_PRE_CUTOFF'] = df['RECENCY_MONTHS_PRE_CUTOFF'].isnull().astype(int)
df['RECENCY_MONTHS_PRE_CUTOFF'] = df['RECENCY_MONTHS_PRE_CUTOFF'].fillna(99).astype(int)

# Limited history pre-cutoff indicator
df['IS_LIMITED_PRE_CUTOFF_HISTORY'] = ((df['TENURE_DAYS_PRE_CUTOFF'] < 90) | (df['PRE_CUTOFF_OBSERVED_MONTHS'] < 4)).astype(int)

print(f"Features engineered successfully. Table shape: {df.shape}")'''),

    nbf.v4.new_markdown_cell('''### 5. Validation Cohort Assignment
We assign each member to an explicit validation cohort:
* `TRAIN`: Enrolled on or before June 30, 2023.
* `VAL`: Enrolled July 1, 2023 through March 31, 2024.
* `EXCLUDED_POST_CUTOFF`: Enrolled after March 31, 2024 (not available to score at cutoff).'''),

    nbf.v4.new_code_cell('''# Assign validation cohorts
df['VALIDATION_COHORT'] = 'EXCLUDED_POST_CUTOFF'
df.loc[df['SIGNUP_DATE'] <= '2023-06-30', 'VALIDATION_COHORT'] = 'TRAIN'
df.loc[(df['SIGNUP_DATE'] > '2023-06-30') & (df['SIGNUP_DATE'] <= '2024-03-31'), 'VALIDATION_COHORT'] = 'VAL'

print("Cohort Distribution:")
print(df['VALIDATION_COHORT'].value_counts())
print("\\nCohort x Churned Cross-Tabulation:")
print(pd.crosstab(df['VALIDATION_COHORT'], df['CHURNED'], margins=True))'''),

    nbf.v4.new_markdown_cell('''### 6. Validation and Dataset Export
We verify row uniqueness, target consistency, and export `phase2_modeling_dataset.csv`.'''),

    nbf.v4.new_code_cell('''# Integrity assertions
assert len(df) == 2600, "Error: Master row count must be exactly 2,600."
assert df['CUSTOMER_ID'].nunique() == 2600, "Error: Duplicate CUSTOMER_IDs detected."
assert df['CHURNED'].value_counts()[1] == 818, "Error: Churned count mismatch."
assert df['CHURNED'].value_counts()[0] == 1782, "Error: Retained count mismatch."

# Export dataset
df.to_csv('phase2_modeling_dataset.csv', index=False)
print("Validation assertions passed. Exported 'phase2_modeling_dataset.csv' successfully.")''')
]

nb.cells = cells
with open('phase2_feature_engineering_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Updated phase2_feature_engineering_notebook.ipynb created successfully.")
