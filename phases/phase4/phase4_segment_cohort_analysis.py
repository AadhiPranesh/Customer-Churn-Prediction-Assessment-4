"""
FreshBasket Customer Churn Prediction
Phase 4: Customer Segment & Signup Cohort Churn Risk Analysis

Generates comprehensive, reconciled segment and cohort churn breakdowns:
1. phase4_segment_cohort_churn_analysis.csv (All segments: Membership Tier, City, Home Store Price Tier, Signup Cohort)
2. phase4_signup_cohort_analysis.csv (Detailed signup cohorts with tenure and churn rates)
3. phase4_signup_cohort_distribution.png (Visual distribution of volume and churn rate across signup cohorts)
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
        PHASE4_OUTPUT,
        PHASE4_CHARTS
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv"
    PHASE4_OUTPUT = PROJECT_ROOT / "phases" / "phase4" / "output"
    PHASE4_CHARTS = PHASE4_OUTPUT / "charts"

PHASE4_OUTPUT.mkdir(parents=True, exist_ok=True)
PHASE4_CHARTS.mkdir(parents=True, exist_ok=True)


def define_signup_cohorts(df):
    """Assign meaningful signup cohort groups based on enrollment date."""
    dates = pd.to_datetime(df['SIGNUP_DATE'])
    
    def assign_cohort(d):
        if d <= pd.Timestamp('2021-12-31'):
            return '2021 Signups (Early Adopters)'
        elif d <= pd.Timestamp('2022-12-31'):
            return '2022 Signups (Mature Members)'
        elif d <= pd.Timestamp('2023-06-30'):
            return '2023 H1 Signups (Pre-Cutoff Train)'
        elif d <= pd.Timestamp('2023-09-30'):
            return '2023 Q3 Signups (Val Cohort: 6-9M Tenure)'
        elif d <= pd.Timestamp('2023-12-31'):
            return '2023 Q4 Signups (Val Cohort: 3-6M Tenure)'
        elif d <= pd.Timestamp('2024-03-31'):
            return '2024 Q1 Signups (Val Cohort: 0-3M Tenure)'
        else:
            return '2024 Q2 Signups (Post-Cutoff Excluded)'

    df['SIGNUP_COHORT_GROUP'] = dates.apply(assign_cohort)
    df['SIGNUP_QUARTER'] = dates.dt.to_period('Q').astype(str)
    return df


def generate_segment_and_cohort_analysis():
    print("=== Generating Customer Segment & Signup Cohort Churn Risk Analysis ===")
    
    df = pd.read_csv(PROCESSED_DATASET_PATH)
    df = define_signup_cohorts(df)
    
    # Analyze eligible population (2,549 customers) and validation cohort (602 customers)
    eligible_df = df[df['VALIDATION_COHORT'] != 'EXCLUDED_POST_CUTOFF'].copy()
    val_df = df[df['VALIDATION_COHORT'] == 'VAL'].copy()
    
    # -------------------------------------------------------------
    # 1. Comprehensive Segment Analysis Table
    # Required columns: Segment_Type, Segment_Name, Customer_Count, Churner_Count, Retained_Count, Observed_Churn_Rate
    # -------------------------------------------------------------
    records = []
    
    # A. Membership Tier (Validation Cohort for evaluation consistency)
    for tier in ['Silver', 'Gold', 'Platinum']:
        sub = val_df[val_df['MEMBERSHIP_TIER'] == tier]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        records.append({
            'Segment_Type': 'Membership_Tier',
            'Segment_Name': tier,
            'Cohort_Scope': 'Validation_Cohort_N602',
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate': rate
        })
        
    # B. City (Validation Cohort)
    for city in sorted(val_df['CITY'].dropna().unique()):
        sub = val_df[val_df['CITY'] == city]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        records.append({
            'Segment_Type': 'City',
            'Segment_Name': city,
            'Cohort_Scope': 'Validation_Cohort_N602',
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate': rate
        })

    # C. Home Store Price Tier (Validation Cohort)
    for store in ['Mainstream', 'Value', 'Upscale', 'Unknown']:
        sub = val_df[val_df['HOME_STORE_PRICE_TIER'] == store]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        records.append({
            'Segment_Type': 'Home_Store_Price_Tier',
            'Segment_Name': store,
            'Cohort_Scope': 'Validation_Cohort_N602',
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate': rate
        })

    # D. Signup Cohort (Validation Cohort)
    val_cohorts = [
        '2023 Q3 Signups (Val Cohort: 6-9M Tenure)',
        '2023 Q4 Signups (Val Cohort: 3-6M Tenure)',
        '2024 Q1 Signups (Val Cohort: 0-3M Tenure)'
    ]
    for sc in val_cohorts:
        sub = val_df[val_df['SIGNUP_COHORT_GROUP'] == sc]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        records.append({
            'Segment_Type': 'Signup_Cohort',
            'Segment_Name': sc,
            'Cohort_Scope': 'Validation_Cohort_N602',
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate': rate
        })

    # Also include the overall population versions for macro-business context
    # Scope 2: All Eligible (N = 2,549)
    for tier in ['Silver', 'Gold', 'Platinum']:
        sub = eligible_df[eligible_df['MEMBERSHIP_TIER'] == tier]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        records.append({
            'Segment_Type': 'Membership_Tier',
            'Segment_Name': tier,
            'Cohort_Scope': 'All_Eligible_N2549',
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate': rate
        })

    for city in sorted(eligible_df['CITY'].dropna().unique()):
        sub = eligible_df[eligible_df['CITY'] == city]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        records.append({
            'Segment_Type': 'City',
            'Segment_Name': city,
            'Cohort_Scope': 'All_Eligible_N2549',
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate': rate
        })

    for store in ['Mainstream', 'Value', 'Upscale', 'Unknown']:
        sub = eligible_df[eligible_df['HOME_STORE_PRICE_TIER'] == store]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        records.append({
            'Segment_Type': 'Home_Store_Price_Tier',
            'Segment_Name': store,
            'Cohort_Scope': 'All_Eligible_N2549',
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate': rate
        })

    for sc in [
        '2021 Signups (Early Adopters)',
        '2022 Signups (Mature Members)',
        '2023 H1 Signups (Pre-Cutoff Train)',
        '2023 Q3 Signups (Val Cohort: 6-9M Tenure)',
        '2023 Q4 Signups (Val Cohort: 3-6M Tenure)',
        '2024 Q1 Signups (Val Cohort: 0-3M Tenure)'
    ]:
        sub = eligible_df[eligible_df['SIGNUP_COHORT_GROUP'] == sc]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        records.append({
            'Segment_Type': 'Signup_Cohort',
            'Segment_Name': sc,
            'Cohort_Scope': 'All_Eligible_N2549',
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate': rate
        })

    # Scope 3: Full Population (N = 2,600)
    for tier in ['Silver', 'Gold', 'Platinum']:
        sub = df[df['MEMBERSHIP_TIER'] == tier]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        records.append({
            'Segment_Type': 'Membership_Tier',
            'Segment_Name': tier,
            'Cohort_Scope': 'Full_Population_N2600',
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate': rate
        })

    for city in sorted(df['CITY'].dropna().unique()):
        sub = df[df['CITY'] == city]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        records.append({
            'Segment_Type': 'City',
            'Segment_Name': city,
            'Cohort_Scope': 'Full_Population_N2600',
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate': rate
        })

    for store in ['Mainstream', 'Value', 'Upscale', 'Unknown']:
        sub = df[df['HOME_STORE_PRICE_TIER'] == store]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        records.append({
            'Segment_Type': 'Home_Store_Price_Tier',
            'Segment_Name': store,
            'Cohort_Scope': 'Full_Population_N2600',
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate': rate
        })

    for sc in [
        '2021 Signups (Early Adopters)',
        '2022 Signups (Mature Members)',
        '2023 H1 Signups (Pre-Cutoff Train)',
        '2023 Q3 Signups (Val Cohort: 6-9M Tenure)',
        '2023 Q4 Signups (Val Cohort: 3-6M Tenure)',
        '2024 Q1 Signups (Val Cohort: 0-3M Tenure)',
        '2024 Q2 Signups (Post-Cutoff Excluded)'
    ]:
        sub = df[df['SIGNUP_COHORT_GROUP'] == sc]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        records.append({
            'Segment_Type': 'Signup_Cohort',
            'Segment_Name': sc,
            'Cohort_Scope': 'Full_Population_N2600',
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate': rate
        })

    seg_df = pd.DataFrame(records)
    seg_csv_path = PHASE4_OUTPUT / "phase4_segment_cohort_churn_analysis.csv"
    seg_df.to_csv(seg_csv_path, index=False)
    print(f"Saved complete segment and cohort analysis to '{seg_csv_path}' ({len(seg_df)} rows).")

    # -------------------------------------------------------------
    # 2. Dedicated Signup Cohort Analysis Table (All 2,600 members)
    # -------------------------------------------------------------
    cohort_summary = []
    cohort_order = [
        '2021 Signups (Early Adopters)',
        '2022 Signups (Mature Members)',
        '2023 H1 Signups (Pre-Cutoff Train)',
        '2023 Q3 Signups (Val Cohort: 6-9M Tenure)',
        '2023 Q4 Signups (Val Cohort: 3-6M Tenure)',
        '2024 Q1 Signups (Val Cohort: 0-3M Tenure)',
        '2024 Q2 Signups (Post-Cutoff Excluded)'
    ]
    
    for cname in cohort_order:
        sub = df[df['SIGNUP_COHORT_GROUP'] == cname]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ret = n - ch
        rate = round(ch / n * 100, 2) if n > 0 else 0.0
        mean_tenure = round(sub['TENURE_DAYS_PRE_CUTOFF'].mean(), 1)
        mean_txns = round(sub['TOTAL_TXNS_PRE_CUTOFF'].mean(), 1)
        mean_spend = round(sub['TOTAL_SPEND_PRE_CUTOFF'].mean(), 2)
        split_role = sub['VALIDATION_COHORT'].iloc[0] if n > 0 else 'UNKNOWN'
        
        cohort_summary.append({
            'Signup_Cohort': cname,
            'Modeling_Split': split_role,
            'Customer_Count': n,
            'Churner_Count': ch,
            'Retained_Count': ret,
            'Observed_Churn_Rate_Pct': rate,
            'Mean_Pre_Cutoff_Tenure_Days': mean_tenure,
            'Mean_Pre_Cutoff_Txns': mean_txns,
            'Mean_Pre_Cutoff_Spend': mean_spend
        })
        
    cohort_df = pd.DataFrame(cohort_summary)
    cohort_csv_path = PHASE4_OUTPUT / "phase4_signup_cohort_analysis.csv"
    cohort_df.to_csv(cohort_csv_path, index=False)
    print(f"Saved signup cohort summary to '{cohort_csv_path}'.")

    # -------------------------------------------------------------
    # 3. High-Resolution Signup Cohort Chart
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax1 = plt.subplots(figsize=(13, 6.5))
    
    # Filter out post-cutoff excluded cohort for main modeling visualization
    plot_cohorts = cohort_df[cohort_df['Modeling_Split'] != 'EXCLUDED_POST_CUTOFF'].copy()
    labels = [
        '2021 Signups\n(Train: Mature)',
        '2022 Signups\n(Train: Mature)',
        '2023 H1 Signups\n(Train: Baseline)',
        '2023 Q3 Signups\n(Val: 6-9M Tenure)',
        '2023 Q4 Signups\n(Val: 3-6M Tenure)',
        '2024 Q1 Signups\n(Val: 0-3M Tenure)'
    ]
    x = np.arange(len(plot_cohorts))
    width = 0.55
    
    # Bar plot for customer volume
    colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#aec7e8', '#aec7e8', '#aec7e8']
    bars = ax1.bar(x, plot_cohorts['Customer_Count'], width, color=colors, alpha=0.75, edgecolor='black', linewidth=0.8, label='Customer Volume (N)')
    ax1.set_ylabel('Number of Loyalty Members', fontsize=12, fontweight='bold', color='#1f77b4')
    ax1.set_ylim(0, 950)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 15, f'N={int(h)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Line plot on secondary axis for observed churn rate
    ax2 = ax1.twinx()
    ax2.plot(x, plot_cohorts['Observed_Churn_Rate_Pct'], color='#d62728', marker='o', linewidth=2.8, markersize=8, label='Observed Churn Rate (%)')
    ax2.set_ylabel('Observed Churn Rate (%)', fontsize=12, fontweight='bold', color='#d62728')
    ax2.set_ylim(0, 50)
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.grid(False)
    
    for i, rate in enumerate(plot_cohorts['Observed_Churn_Rate_Pct']):
        ax2.annotate(f'{rate:.1f}%', (x[i], rate + 1.8), ha='center', fontsize=10, fontweight='bold', color='#d62728',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#d62728', alpha=0.85))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax1.set_title('FreshBasket Customer Volume & Observed Churn Rate Across Signup Cohorts\n(Temporal Progression from Mature Base to Recent Joiners)', fontsize=13, fontweight='bold', pad=15)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)
    
    chart_path = PHASE4_CHARTS / "phase4_signup_cohort_distribution.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved signup cohort chart to '{chart_path}'.")

    # Reconciliation check
    val_sub = seg_df[(seg_df['Cohort_Scope'] == 'Validation_Cohort_N602') & (seg_df['Segment_Type'] == 'Signup_Cohort')]
    val_n = val_sub['Customer_Count'].sum()
    val_ch = val_sub['Churner_Count'].sum()
    assert val_n == 602, f"Reconciliation error: expected 602 validation customers, got {val_n}"
    assert val_ch == 85, f"Reconciliation error: expected 85 validation churners, got {val_ch}"
    print("RECONCILIATION SUCCESS: Validation cohort signup breakdown perfectly equals 602 customers and 85 churners.")
    return True


if __name__ == '__main__':
    generate_segment_and_cohort_analysis()
