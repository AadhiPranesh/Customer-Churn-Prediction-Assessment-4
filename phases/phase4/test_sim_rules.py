import os
from pathlib import Path
import pandas as pd

try:
    from phases.paths import PROJECT_ROOT, VALIDATION_PREDS_PATH, PROCESSED_DATASET_PATH
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    VALIDATION_PREDS_PATH = PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions.csv"
    PROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv"

def _find(p1, p2):
    if p1.exists(): return p1
    if p2.exists(): return p2
    return p1

preds_path = _find(VALIDATION_PREDS_PATH, PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions.csv")
dataset_path = _find(PROCESSED_DATASET_PATH, PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv")


df_preds = pd.read_csv(preds_path)
df_feat = pd.read_csv(dataset_path)

merged = pd.merge(df_preds, df_feat, on=['CUSTOMER_ID', 'SIGNUP_DATE', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER', 'CHURNED'])

at_risk = merged[merged['RF_PRED_LABEL'] == 1]
print(f"Total Validation Customers: {len(merged)}")
print(f"Total At-Risk Customers (RF Pred Churn, prob >= 0.50): {len(at_risk)}")
print(f"Actual churners among At-Risk: {at_risk['CHURNED'].sum()} ({at_risk['CHURNED'].mean()*100:.1f}%)")

print("\n--- Support & Complaint Distribution among At-Risk (N=105) ---")
has_ticket = at_risk['TOTAL_SUPPORT_TICKETS_PRE_CUTOFF'] > 0
has_complaint = at_risk['COMPLAINT_RECORDED_PRE_CUTOFF'] == 1
scenario1 = has_ticket | has_complaint
print(f"Support Tickets > 0:       {has_ticket.sum()}")
print(f"Complaint Recorded == 1:   {has_complaint.sum()}")
print(f"Scenario 1 (Ticket or Complaint): {scenario1.sum()}")

print("\n--- Recency Distribution among At-Risk (N=105) ---")
print(at_risk['RECENCY_MONTHS_PRE_CUTOFF'].value_counts().sort_index())
# For grocery, dormancy is >= 2 months (60+ days without a grocery visit)
rec_ge_2 = at_risk['RECENCY_MONTHS_PRE_CUTOFF'] >= 2
rec_ge_3 = at_risk['RECENCY_MONTHS_PRE_CUTOFF'] >= 3
print(f"Recency >= 2 months:       {rec_ge_2.sum()}")
print(f"Recency >= 3 months:       {rec_ge_3.sum()}")

print("\n--- Digital Engagement Distribution among At-Risk (N=105) ---")
low_app = at_risk['AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF'] == 0
low_email = at_risk['AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF'] == 0
scenario3_strict = low_app | low_email
scenario3_both_zero = low_app & low_email
print(f"App Sessions == 0:         {low_app.sum()}")
print(f"Email Opens == 0:          {low_email.sum()}")
print(f"Scenario 3 (Zero App OR Zero Email): {scenario3_strict.sum()}")
print(f"Scenario 3 (Zero App AND Zero Email): {scenario3_both_zero.sum()}")

# Overlap between scenarios
scenario2 = rec_ge_2
print("\n--- Overlap Analysis (Scenario 1, 2, 3) ---")
print(f"Scenario 1 (Support/Complaint):    {scenario1.sum()}")
print(f"Scenario 2 (Recency >= 2 months):  {scenario2.sum()}")
print(f"Scenario 3 (Zero App or Email):    {scenario3_strict.sum()}")

# Detailed Venn overlap
s1_only = scenario1 & (~scenario2) & (~scenario3_strict)
s2_only = scenario2 & (~scenario1) & (~scenario3_strict)
s3_only = scenario3_strict & (~scenario1) & (~scenario2)
s1_and_s2 = scenario1 & scenario2 & (~scenario3_strict)
s1_and_s3 = scenario1 & scenario3_strict & (~scenario2)
s2_and_s3 = scenario2 & scenario3_strict & (~scenario1)
all_three = scenario1 & scenario2 & scenario3_strict
any_scenario = scenario1 | scenario2 | scenario3_strict
none_targeted = (~any_scenario)

print(f"\nUnique Customers Targeted by at least one scenario: {any_scenario.sum()} of {len(at_risk)} ({any_scenario.sum()/len(at_risk)*100:.1f}%)")
print(f"Not targeted by any scenario: {none_targeted.sum()}")
print(f"Overlap breakdown:")
print(f"  • Scenario 1 Only:        {s1_only.sum()}")
print(f"  • Scenario 2 Only:        {s2_only.sum()}")
print(f"  • Scenario 3 Only:        {s3_only.sum()}")
print(f"  • S1 & S2:                {s1_and_s2.sum()}")
print(f"  • S1 & S3:                {s1_and_s3.sum()}")
print(f"  • S2 & S3:                {s2_and_s3.sum()}")
print(f"  • All Three (S1 & S2 & S3): {all_three.sum()}")
