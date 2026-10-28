"""
FreshBasket Customer Churn Prediction
Phase 4 — Step 3: Actionable Retention Strategy Simulation
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Safe path resolution
try:
    from phases.paths import (
        PROJECT_ROOT,
        PROCESSED_DATASET_PATH,
        VALIDATION_PREDS_PATH,
        PHASE4_OUTPUT as OUTPUTS_ANALYSIS,
        PHASE4_CHARTS as CHARTS_DIR
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv"
    VALIDATION_PREDS_PATH = PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions.csv"
    OUTPUTS_ANALYSIS = PROJECT_ROOT / "phases" / "phase4" / "output"
    CHARTS_DIR = OUTPUTS_ANALYSIS / "charts"

OUTPUTS_ANALYSIS.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)




def _find_file(preferred, fallback):
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    return preferred


def run_retention_simulation():
    # 1. Ingest Data
    preds_path = _find_file(VALIDATION_PREDS_PATH, PROJECT_ROOT / "phase3_validation_predictions.csv")
    master_path = _find_file(PROCESSED_DATASET_PATH, PROJECT_ROOT / "phase2_modeling_dataset.csv")

    df_preds = pd.read_csv(preds_path)
    df_feat = pd.read_csv(master_path)

    merged = pd.merge(
        df_preds,
        df_feat,
        on=['CUSTOMER_ID', 'SIGNUP_DATE', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER', 'CHURNED']
    )

    # Filter At-Risk Cohort (Threshold = 0.50, RF_PRED_LABEL == 1)
    at_risk = merged[merged['RF_PRED_LABEL'] == 1].copy()

    # Define Scenario Targeting Rules
    # Scenario 1: Support / Friction Follow-up
    at_risk['S1_SUPPORT'] = (at_risk['TOTAL_SUPPORT_TICKETS_PRE_CUTOFF'] > 0) | (at_risk['COMPLAINT_RECORDED_PRE_CUTOFF'] == 1)

    # Scenario 2: High-Dormancy Reactivation
    at_risk['S2_REACTIVATION'] = (at_risk['RECENCY_MONTHS_PRE_CUTOFF'] >= 2)

    # Scenario 3: Digital Re-engagement
    at_risk['S3_DIGITAL'] = (at_risk['AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF'] == 0) | (at_risk['AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF'] == 0)

    # Combined Deduplicated Strategy
    at_risk['COMBINED_TARGETED'] = at_risk['S1_SUPPORT'] | at_risk['S2_REACTIVATION'] | at_risk['S3_DIGITAL']

    # 2. Evaluate Scenarios
    scenarios = [
        ('S1', 'Scenario 1: Support Follow-up', at_risk['S1_SUPPORT'], 'Support Tickets > 0 OR Complaint Recorded == 1', 'Direct phone concierge resolution'),
        ('S2', 'Scenario 2: Reactivation Offer', at_risk['S2_REACTIVATION'], 'Purchase Recency >= 2 Calendar Months', 'Win-back coupon booklet via direct mail/app'),
        ('S3', 'Scenario 3: Digital Re-engagement', at_risk['S3_DIGITAL'], 'Zero Pre-Cutoff App Sessions OR Zero Email Opens', 'App onboarding push + personalized welcome sequence'),
        ('COMBINED', 'Combined Program (Deduplicated Unique Customers)', at_risk['COMBINED_TARGETED'], 'Targeted by at least one of Scenario 1, 2, or 3', 'Unified multi-channel retention playbook')
    ]

    sim_results = []
    efficacies = [0.10, 0.20, 0.30]

    for sc_id, sc_name, mask, rule_desc, action_desc in scenarios:
        sub = at_risk[mask]
        n_targeted = len(sub)
        n_churn_targeted = int(sub['CHURNED'].sum())
        n_fp_targeted = n_targeted - n_churn_targeted
        
        prec = round(n_churn_targeted / n_targeted * 100, 2) if n_targeted > 0 else 0.0
        coverage = round(n_churn_targeted / 85 * 100, 2)
        mean_rf_prob = round(sub['RF_CHURN_PROBABILITY'].mean() * 100, 2) if n_targeted > 0 else 0.0
        
        saved_10 = round(n_churn_targeted * 0.10, 1)
        saved_20 = round(n_churn_targeted * 0.20, 1)
        saved_30 = round(n_churn_targeted * 0.30, 1)
        
        sim_results.append({
            'Scenario_ID': sc_id,
            'Scenario_Name': sc_name,
            'Eligibility_Rule': rule_desc,
            'Intervention_Mechanism': action_desc,
            'Targeted_Customers_N': n_targeted,
            'Actual_Churners_Targeted': n_churn_targeted,
            'Actual_Retained_Targeted_FP': n_fp_targeted,
            'Targeting_Precision_Pct': prec,
            'Validation_Churn_Coverage_Pct': coverage,
            'Mean_RF_Churn_Probability_Pct': mean_rf_prob,
            'Hypothetical_Saved_Efficacy_10pct': saved_10,
            'Hypothetical_Saved_Efficacy_20pct': saved_20,
            'Hypothetical_Saved_Efficacy_30pct': saved_30
        })

    df_sim = pd.DataFrame(sim_results)
    df_sim.to_csv(OUTPUTS_ANALYSIS / 'phase4_step3_retention_scenario_results.csv', index=False)

    # Customer targeting roster
    roster_cols = [
        'CUSTOMER_ID', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER', 'CITY',
        'RF_CHURN_PROBABILITY', 'RF_PRED_LABEL', 'LR_CHURN_PROBABILITY', 'CHURNED',
        'RECENCY_MONTHS_PRE_CUTOFF', 'TXNS_LAST_3M', 'SPEND_LAST_3M',
        'TOTAL_SUPPORT_TICKETS_PRE_CUTOFF', 'COMPLAINT_RECORDED_PRE_CUTOFF',
        'AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF', 'AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF',
        'S1_SUPPORT', 'S2_REACTIVATION', 'S3_DIGITAL', 'COMBINED_TARGETED'
    ]
    at_risk_roster = at_risk[roster_cols].copy().sort_values(by='RF_CHURN_PROBABILITY', ascending=False)
    at_risk_roster.to_csv(OUTPUTS_ANALYSIS / 'phase4_step3_customer_targeting_roster.csv', index=False)

    # Overlap analysis
    s1 = at_risk['S1_SUPPORT']
    s2 = at_risk['S2_REACTIVATION']
    s3 = at_risk['S3_DIGITAL']

    overlap_data = [
        {'Segment': 'Scenario 1 Only (Support)', 'Count': int((s1 & ~s2 & ~s3).sum()), 'Actual_Churners': int(at_risk[s1 & ~s2 & ~s3]['CHURNED'].sum())},
        {'Segment': 'Scenario 2 Only (Recency)', 'Count': int((s2 & ~s1 & ~s3).sum()), 'Actual_Churners': int(at_risk[s2 & ~s1 & ~s3]['CHURNED'].sum())},
        {'Segment': 'Scenario 3 Only (Digital)', 'Count': int((s3 & ~s1 & ~s2).sum()), 'Actual_Churners': int(at_risk[s3 & ~s1 & ~s2]['CHURNED'].sum())},
        {'Segment': 'S1 and S2 Only',           'Count': int((s1 & s2 & ~s3).sum()),  'Actual_Churners': int(at_risk[s1 & s2 & ~s3]['CHURNED'].sum())},
        {'Segment': 'S1 and S3 Only',           'Count': int((s1 & s3 & ~s2).sum()),  'Actual_Churners': int(at_risk[s1 & s3 & ~s2]['CHURNED'].sum())},
        {'Segment': 'S2 and S3 Only',           'Count': int((s2 & s3 & ~s1).sum()),  'Actual_Churners': int(at_risk[s2 & s3 & ~s1]['CHURNED'].sum())},
        {'Segment': 'All Three Scenarios',       'Count': int((s1 & s2 & s3).sum()),   'Actual_Churners': int(at_risk[s1 & s2 & s3]['CHURNED'].sum())},
        {'Segment': 'Total Unique Targeted',    'Count': int((s1 | s2 | s3).sum()),   'Actual_Churners': int(at_risk[s1 | s2 | s3]['CHURNED'].sum())},
        {'Segment': 'Not Targeted by Any S1-S3','Count': int((~s1 & ~s2 & ~s3).sum()),'Actual_Churners': int(at_risk[~s1 & ~s2 & ~s3]['CHURNED'].sum())}
    ]
    df_overlap = pd.DataFrame(overlap_data)
    df_overlap.to_csv(OUTPUTS_ANALYSIS / 'phase4_step3_scenario_overlap.csv', index=False)

    # Visualizations
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax1 = axes[0]
    sc_names = ['S1: Support\nFollow-up', 'S2: Reactivation\nOffer', 'S3: Digital\nRe-engagement', 'Combined\n(Unique)']
    targeted = df_sim['Targeted_Customers_N']
    churners = df_sim['Actual_Churners_Targeted']

    x = np.arange(len(sc_names))
    w = 0.35
    r1 = ax1.bar(x - w/2, targeted, w, label='Total Targeted Customers (N)', color='#1f77b4', alpha=0.85)
    r2 = ax1.bar(x + w/2, churners, w, label='Actual Churners Included (TP)', color='#2ca02c', alpha=0.85)

    ax1.set_ylabel('Customer Count', fontsize=12, fontweight='bold')
    ax1.set_title('Targeting Volume vs. Genuine Churners Identified', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sc_names, fontsize=10, fontweight='bold')
    ax1.legend()

    for rect in r1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 1, f'{int(h)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in r2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 1, f'{int(h)}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2ca02c')

    ax2 = axes[1]
    w2 = 0.25
    r_10 = ax2.bar(x - w2, df_sim['Hypothetical_Saved_Efficacy_10pct'], w2, label='10% Efficacy Rate', color='#ffbb78', alpha=0.9)
    r_20 = ax2.bar(x, df_sim['Hypothetical_Saved_Efficacy_20pct'], w2, label='20% Efficacy Rate', color='#ff7f0e', alpha=0.9)
    r_30 = ax2.bar(x + w2, df_sim['Hypothetical_Saved_Efficacy_30pct'], w2, label='30% Efficacy Rate', color='#d62728', alpha=0.9)

    ax2.set_ylabel('Hypothetical Churners Saved', fontsize=12, fontweight='bold')
    ax2.set_title('Sensitivity Analysis: Churners Retained across Efficacy Assumed', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sc_names, fontsize=10, fontweight='bold')
    ax2.legend()

    for rect in r_10:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for rect in r_20:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for rect in r_30:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'phase4_retention_simulation.png', dpi=300)
    plt.close()

    print("\n=== Phase 4 Step 3 Simulation Completed Successfully ===")


if __name__ == '__main__':
    run_retention_simulation()
