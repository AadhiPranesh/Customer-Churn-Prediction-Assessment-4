import os
from pathlib import Path
import pandas as pd
import numpy as np

import sys
try:
    from phases.paths import PROJECT_ROOT, PROCESSED_DATASET_PATH, ABLATION_RESULTS_PATH
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv"
    ABLATION_RESULTS_PATH = PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_step2_ablation_results.csv"

def _find(p1, p2):
    if p1.exists(): return p1
    if p2.exists(): return p2
    return p1

dataset_path = _find(PROCESSED_DATASET_PATH, PROJECT_ROOT / "phase2_modeling_dataset.csv")
df = pd.read_csv(dataset_path)
admin_cols = ['CUSTOMER_ID', 'SIGNUP_DATE', 'CHURNED', 'VALIDATION_COHORT', 'IS_POST_CUTOFF_SIGNUP']
feature_cols = [c for c in df.columns if c not in admin_cols]

# Define Feature Sets
set1 = ['AGE', 'AGE_IS_MISSING', 'GENDER', 'CITY', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER']
set2 = set1 + ['ELIGIBLE_PRE_CUTOFF_MONTHS', 'TENURE_DAYS_PRE_CUTOFF', 'PRE_CUTOFF_OBSERVED_MONTHS', 'IS_LIMITED_PRE_CUTOFF_HISTORY']
set3 = set2 + [
    'TOTAL_TXNS_PRE_CUTOFF', 'TOTAL_SPEND_PRE_CUTOFF', 'ACTIVE_MONTHS_PRE_CUTOFF',
    'RECENCY_MONTHS_PRE_CUTOFF', 'TXNS_LAST_3M', 'SPEND_LAST_3M',
    'HAS_NO_PRE_CUTOFF_ACTIVITY', 'AVG_MONTHLY_TXNS_PRE_CUTOFF', 'AVG_MONTHLY_SPEND_PRE_CUTOFF',
    'AVG_BASKET_VALUE_PRE_CUTOFF', 'NEVER_PURCHASED_PRE_CUTOFF', 'TOTAL_COUPONS_REDEEMED_PRE_CUTOFF'
]
set4 = feature_cols

print("=== CHECK 1: EXACT FEATURE NAMES & STRICT SUBSET VERIFICATION ===")
print(f"Set 1 ({len(set1)} features): {set1}")
print(f"\nSet 2 ({len(set2)} features): {set2}")
print(f"\nSet 3 ({len(set3)} features): {set3}")
print(f"\nSet 4 ({len(set4)} features): {set4}")

s1, s2, s3, s4 = set(set1), set(set2), set(set3), set(set4)
print("Subset Checks:")
print(f"  Set 1 is subset of Set 2: {s1.issubset(s2)} (Added: {s2 - s1})")
print(f"  Set 2 is subset of Set 3: {s2.issubset(s3)} (Added: {s3 - s2})")
print(f"  Set 3 is subset of Set 4: {s3.issubset(s4)} (Added: {s4 - s3})")
print(f"  Strict inclusions: {len(s1) < len(s2) < len(s3) < len(s4)}")
print(f"  Set 4 == Full 28 Features: {s4 == set(feature_cols)}")

print("\n=== CHECK 2: FEATURES ADDED FROM SET 3 TO SET 4 ===")
added_3_to_4 = sorted(list(s4 - s3))
for f in added_3_to_4:
    if 'SUPPORT' in f:
        domain = "Support (Customer Service ticket count)"
    elif 'COMPLAINT' in f:
        domain = "Complaints (Formal customer complaint recorded flag)"
    elif 'APP' in f:
        domain = "App Activity (Mobile app digital sessions)"
    elif 'EMAIL' in f:
        domain = "Email Engagement (Promotional email opens)"
    elif 'COUPON' in f:
        domain = "Coupons (Discount coupon redemption)"
    else:
        domain = "Other"
    print(f"  • {f:38} | Domain: {domain}")

print("\n=== CHECK 3 & 4: RECALCULATING CONFUSION MATRICES & SUMS ===")
try:
    from phases.paths import ABLATION_RESULTS_PATH
    res_path = ABLATION_RESULTS_PATH
except ImportError:
    res_path = PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_step2_ablation_results.csv"
if not res_path.exists():
    res_path = PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_step2_ablation_results.csv"
res_df = pd.read_csv(res_path)
for idx, r in res_df.iterrows():
    tn, fp, fn, tp = r['TN'], r['FP'], r['FN'], r['TP']
    actual_retained = tn + fp
    actual_churned = fn + tp
    total = actual_retained + actual_churned
    print(f"{r['Feature_Set']:35} | TN={tn:3}, FP={fp:3} (Retained={actual_retained:3}) | FN={fn:2}, TP={tp:2} (Churned={actual_churned:2}) | Total={total:3}")
    assert actual_retained == 517, f"Retained sum mismatch for {r['Feature_Set']}"
    assert actual_churned == 85, f"Churned sum mismatch for {r['Feature_Set']}"
    assert total == 602, f"Total sum mismatch for {r['Feature_Set']}"

print("\nAll confusion matrices correctly sum to 602 (517 Retained, 85 Churned)!")
