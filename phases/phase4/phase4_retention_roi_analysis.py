"""
FreshBasket Customer Churn Prediction
Phase 4: Retention Campaign Financial Cost & ROI Sensitivity Analysis

Performs financial cost-benefit and ROI sensitivity modeling:
- Uses defensible, editable illustrative business assumptions derived from grocery retail benchmarks
- Evaluates scenarios across 10%, 20%, and 30% intervention efficacy rates
- Explicitly separates actual verified validation figures from hypothetical financial outcomes
- Calculates:
    * Total Campaign Cost = Targeted Customers * Intervention Cost per Customer
    * Expected Retained Customers = True Churners Targeted * Efficacy
    * Expected Benefit = Expected Retained Customers * Value per Retained Customer
    * Net Benefit = Expected Benefit - Total Campaign Cost
    * ROI (%) = (Expected Benefit - Total Campaign Cost) / Total Campaign Cost * 100%
- Exports:
    1. phases/phase4/output/phase4_roi_assumptions.csv
    2. phases/phase4/output/phase4_retention_financial_roi_sensitivity.csv
    3. phases/phase4/output/charts/phase4_retention_roi_sensitivity.png
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
        PHASE4_OUTPUT,
        PHASE4_CHARTS
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PHASE4_OUTPUT = PROJECT_ROOT / "phases" / "phase4" / "output"
    PHASE4_CHARTS = PHASE4_OUTPUT / "charts"

PHASE4_OUTPUT.mkdir(parents=True, exist_ok=True)
PHASE4_CHARTS.mkdir(parents=True, exist_ok=True)

# Defensible illustrative financial assumptions
# (Derivation: Average monthly spend in FreshBasket is ~$115/mo -> $1,380 annual spend.
# At ~18% retail gross margin, customer annual contribution margin is ~$250/year.)
DEFAULT_ASSUMPTIONS = {
    'S1': {
        'name': 'Scenario 1: Support Follow-up',
        'cost_per_cust': 15.00,
        'cost_rationale': 'Direct phone outreach by customer care agent + $10 appeasement voucher',
        'targeted_n': 54,
        'actual_churners': 45
    },
    'S2': {
        'name': 'Scenario 2: Reactivation Offer',
        'cost_per_cust': 8.00,
        'cost_rationale': 'Targeted physical mailer + high-value digital basket discount booklet ($15 off $75)',
        'targeted_n': 55,
        'actual_churners': 52
    },
    'S3': {
        'name': 'Scenario 3: Digital Re-engagement',
        'cost_per_cust': 2.00,
        'cost_rationale': 'Push notification sequence + automated personalized onboarding email nudge series',
        'targeted_n': 51,
        'actual_churners': 35
    },
    'COMBINED': {
        'name': 'Combined Program (Deduplicated Unique)',
        'cost_per_cust': 10.00,
        'cost_rationale': 'Blended weighted channel execution across phone, mail, and digital streams',
        'targeted_n': 91,
        'actual_churners': 67
    }
}

CUSTOMER_LTV_VALUE = 250.00  # Expected annual gross profit contribution of retained customer
EFFICACIES = [0.10, 0.20, 0.30]


def run_roi_analysis():
    print("=== Generating Retention Campaign Cost & ROI Sensitivity Analysis ===")
    
    # 1. Export Assumption Table
    assumption_rows = []
    for sc_id, data in DEFAULT_ASSUMPTIONS.items():
        assumption_rows.append({
            'Scenario_ID': sc_id,
            'Scenario_Name': data['name'],
            'Targeted_Population_N': data['targeted_n'],
            'True_Churners_Targeted': data['actual_churners'],
            'Intervention_Cost_Per_Customer_USD': data['cost_per_cust'],
            'Assumed_Retained_Customer_Value_USD': CUSTOMER_LTV_VALUE,
            'Evaluated_Efficacy_Scenarios': '10%, 20%, 30%',
            'Source_Rationale': data['cost_rationale'],
            'Assumption_Type': 'Illustrative / Defensible Benchmark'
        })
    assumption_df = pd.DataFrame(assumption_rows)
    assumption_csv = PHASE4_OUTPUT / "phase4_roi_assumptions.csv"
    assumption_df.to_csv(assumption_csv, index=False)
    print(f"Saved assumption table to '{assumption_csv}'.")

    # 2. Compute Financial Sensitivity Matrix
    roi_rows = []
    for sc_id, data in DEFAULT_ASSUMPTIONS.items():
        n_targeted = data['targeted_n']
        churners_tp = data['actual_churners']
        unit_cost = data['cost_per_cust']
        campaign_cost = n_targeted * unit_cost
        
        for eff in EFFICACIES:
            eff_pct_label = f"{int(eff*100)}%"
            expected_retained = round(churners_tp * eff, 2)
            expected_benefit = round(expected_retained * CUSTOMER_LTV_VALUE, 2)
            net_benefit = round(expected_benefit - campaign_cost, 2)
            
            if campaign_cost > 0:
                roi_pct = round((net_benefit / campaign_cost) * 100.0, 2)
            else:
                roi_pct = 0.0
                
            roi_rows.append({
                'Scenario_ID': sc_id,
                'Scenario_Name': data['name'],
                'Efficacy_Scenario': eff_pct_label,
                'Efficacy_Rate': eff,
                'Targeted_Customers_N': n_targeted,
                'Actual_Churners_Targeted_TP': churners_tp,
                'Expected_Retained_Customers': expected_retained,
                'Unit_Intervention_Cost_USD': unit_cost,
                'Total_Campaign_Cost_USD': round(campaign_cost, 2),
                'Customer_Value_USD': CUSTOMER_LTV_VALUE,
                'Expected_Gross_Benefit_USD': expected_benefit,
                'Expected_Net_Benefit_USD': net_benefit,
                'ROI_Percentage': roi_pct,
                'Financial_Output_Status': 'HYPOTHETICAL_ILLUSTRATIVE'
            })

    roi_df = pd.DataFrame(roi_rows)
    roi_csv = PHASE4_OUTPUT / "phase4_retention_financial_roi_sensitivity.csv"
    roi_df.to_csv(roi_csv, index=False)
    print(f"Saved financial ROI sensitivity matrix to '{roi_csv}' ({len(roi_df)} rows).")

    # 3. High-Resolution ROI and Net Benefit Chart
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sc_labels = ['S1: Support\nFollow-up', 'S2: Reactivation\nOffer', 'S3: Digital\nRe-engagement', 'Combined\nProgram']
    sc_ids = ['S1', 'S2', 'S3', 'COMBINED']
    x = np.arange(len(sc_labels))
    w = 0.25

    # Panel 1: Expected Net Benefit ($)
    net_10 = [roi_df[(roi_df['Scenario_ID'] == s) & (roi_df['Efficacy_Rate'] == 0.10)]['Expected_Net_Benefit_USD'].values[0] for s in sc_ids]
    net_20 = [roi_df[(roi_df['Scenario_ID'] == s) & (roi_df['Efficacy_Rate'] == 0.20)]['Expected_Net_Benefit_USD'].values[0] for s in sc_ids]
    net_30 = [roi_df[(roi_df['Scenario_ID'] == s) & (roi_df['Efficacy_Rate'] == 0.30)]['Expected_Net_Benefit_USD'].values[0] for s in sc_ids]

    r1_10 = ax1.bar(x - w, net_10, w, label='10% Efficacy', color='#74c476', alpha=0.9, edgecolor='black', linewidth=0.6)
    r1_20 = ax1.bar(x, net_20, w, label='20% Efficacy', color='#31a354', alpha=0.9, edgecolor='black', linewidth=0.6)
    r1_30 = ax1.bar(x + w, net_30, w, label='30% Efficacy', color='#006d2c', alpha=0.9, edgecolor='black', linewidth=0.6)

    ax1.set_ylabel('Expected Net Benefit ($ USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Illustrative Net Financial Benefit by Scenario & Efficacy\n(Net Benefit = Expected Saved Value - Campaign Cost)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sc_labels, fontsize=10, fontweight='bold')
    ax1.legend(title='Efficacy Rate')

    for rects in [r1_10, r1_20, r1_30]:
        for rect in rects:
            h = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width()/2., h + 50, f'${int(h)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Panel 2: ROI Percentage (%)
    roi_10 = [roi_df[(roi_df['Scenario_ID'] == s) & (roi_df['Efficacy_Rate'] == 0.10)]['ROI_Percentage'].values[0] for s in sc_ids]
    roi_20 = [roi_df[(roi_df['Scenario_ID'] == s) & (roi_df['Efficacy_Rate'] == 0.20)]['ROI_Percentage'].values[0] for s in sc_ids]
    roi_30 = [roi_df[(roi_df['Scenario_ID'] == s) & (roi_df['Efficacy_Rate'] == 0.30)]['ROI_Percentage'].values[0] for s in sc_ids]

    r2_10 = ax2.bar(x - w, roi_10, w, label='10% Efficacy', color='#9ecae1', alpha=0.9, edgecolor='black', linewidth=0.6)
    r2_20 = ax2.bar(x, roi_20, w, label='20% Efficacy', color='#4292c6', alpha=0.9, edgecolor='black', linewidth=0.6)
    r2_30 = ax2.bar(x + w, roi_30, w, label='30% Efficacy', color='#08519c', alpha=0.9, edgecolor='black', linewidth=0.6)

    ax2.set_ylabel('Expected ROI (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Illustrative Campaign ROI (%) by Scenario & Efficacy\n(ROI = Net Benefit / Campaign Cost * 100%)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sc_labels, fontsize=10, fontweight='bold')
    ax2.legend(title='Efficacy Rate')

    for rects in [r2_10, r2_20, r2_30]:
        for rect in rects:
            h = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width()/2., h + 30, f'{int(h)}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    chart_path = PHASE4_CHARTS / "phase4_retention_roi_sensitivity.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved ROI sensitivity visualization to '{chart_path}'.")

    # Arithmetic sanity verification
    for _, row in roi_df.iterrows():
        calc_cost = row['Targeted_Customers_N'] * row['Unit_Intervention_Cost_USD']
        np.testing.assert_almost_equal(calc_cost, row['Total_Campaign_Cost_USD'], decimal=2)
        calc_benefit = row['Expected_Retained_Customers'] * row['Customer_Value_USD']
        np.testing.assert_almost_equal(calc_benefit, row['Expected_Gross_Benefit_USD'], decimal=2)
        calc_net = calc_benefit - calc_cost
        np.testing.assert_almost_equal(calc_net, row['Expected_Net_Benefit_USD'], decimal=2)
        calc_roi = (calc_net / calc_cost) * 100.0
        np.testing.assert_almost_equal(calc_roi, row['ROI_Percentage'], decimal=2)

    print("ARITHMETIC SANITY SUCCESS: All campaign costs, gross benefits, net benefits, and ROI calculations pass 100%.")
    return True


if __name__ == '__main__':
    run_roi_analysis()
