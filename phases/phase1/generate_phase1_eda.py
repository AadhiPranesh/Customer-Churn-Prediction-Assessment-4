"""
FreshBasket Customer Churn Prediction
Phase 1: Exploratory Data Analysis & Visualization Generator

Generates all 7 standardized Phase 1 exploratory charts and exports them
to outputs/eda/charts/.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Safe path resolution
try:
    from phases.paths import (
        PROJECT_ROOT,
        PHASE1_CHARTS as CHARTS_DIR,
        CLEANED_CUSTOMER_PROFILE_PATH,
        CLEANED_CHURN_LABEL_PATH,
        CLEANED_MONTHLY_ACTIVITY_PATH
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PHASE1_OUTPUT = PROJECT_ROOT / "phases" / "phase1" / "output"
    CHARTS_DIR = PHASE1_OUTPUT / "charts"
    CLEANED_CUSTOMER_PROFILE_PATH = PROJECT_ROOT / "data" / "interim" / "cleaned_fb_customer_profile.csv"
    CLEANED_CHURN_LABEL_PATH = PROJECT_ROOT / "data" / "interim" / "cleaned_fb_churn_label.csv"
    CLEANED_MONTHLY_ACTIVITY_PATH = PROJECT_ROOT / "data" / "interim" / "cleaned_fb_monthly_activity.csv"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)



def _find_data_file(canonical, alternatives):
    if canonical.exists():
        return canonical
    for alt in alternatives:
        if alt.exists():
            return alt
    return canonical


def generate_eda_charts():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 150

    cust_path = _find_data_file(
        CLEANED_CUSTOMER_PROFILE_PATH,
        [
            PROJECT_ROOT / "outputs/eda/cleaned_data/cleaned_fb_customer_profile.csv",
            PROJECT_ROOT / "archive/phase_outputs_legacy/phase_1_outputs/cleaned_data/cleaned_fb_customer_profile.csv"
        ]
    )
    churn_path = _find_data_file(
        CLEANED_CHURN_LABEL_PATH,
        [
            PROJECT_ROOT / "outputs/eda/cleaned_data/cleaned_fb_churn_label.csv",
            PROJECT_ROOT / "archive/phase_outputs_legacy/phase_1_outputs/cleaned_data/cleaned_fb_churn_label.csv"
        ]
    )
    act_path = _find_data_file(
        CLEANED_MONTHLY_ACTIVITY_PATH,
        [
            PROJECT_ROOT / "outputs/eda/cleaned_data/cleaned_fb_monthly_activity.csv",
            PROJECT_ROOT / "archive/phase_outputs_legacy/phase_1_outputs/cleaned_data/cleaned_fb_monthly_activity.csv"
        ]
    )

    cust = pd.read_csv(cust_path)
    churn = pd.read_csv(churn_path)
    act = pd.read_csv(act_path)

    act['MONTH'] = pd.to_datetime(act['MONTH'])
    cust['SIGNUP_DATE'] = pd.to_datetime(cust['SIGNUP_DATE'])

    df_cust = cust.merge(churn, on='CUSTOMER_ID', how='inner')

    def save_fig(filename):
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / filename)
        plt.close()

    # Chart 1: Churn Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df_cust['CHURNED'].value_counts().sort_index()
    pcts = counts / len(df_cust) * 100
    colors = ['#2b5c8f', '#d95f02']
    bars = ax.bar(['Retained (0)', 'Churned (1)'], counts, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    for bar, count, pct in zip(bars, counts, pcts):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 25, f'{count:,} ({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('FreshBasket Loyalty Program: Churn Distribution (April-June 2024)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Loyalty Members', fontsize=11)
    ax.set_ylim(0, 2100)
    save_fig('01_churn_distribution.png')

    # Chart 2: Missing Values Summary
    fig, ax = plt.subplots(figsize=(8, 4.5))
    missing_data = {
        'Home Store Price Tier': cust['HOME_STORE_PRICE_TIER'].isnull().sum() if 'HOME_STORE_PRICE_TIER' in cust.columns else 0,
        'Last Purchase Date (Churn)': churn['LAST_PURCHASE_DATE'].isnull().sum(),
        'Member Age': cust['AGE'].isnull().sum(),
        'Member Gender': cust['GENDER'].isnull().sum() if 'GENDER' in cust.columns else 0
    }
    s_missing = pd.Series(missing_data).sort_values(ascending=True)
    s_pct = s_missing / 2600 * 100
    bars = ax.barh(s_missing.index, s_missing.values, color='#4575b4', edgecolor='black', height=0.55)
    for bar, val, pct in zip(bars, s_missing.values, s_pct.values):
        ax.text(val + 3, bar.get_y() + bar.get_height()/2.0, f'{val} ({pct:.1f}%)', ha='left', va='center', fontsize=10, fontweight='bold')
    ax.set_title('Missing Values Identified Across Dataset (Out of 2,600 Members)', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Missing Records', fontsize=11)
    ax.set_xlim(0, 160)
    save_fig('02_missing_values_summary.png')

    # Chart 3: Monthly Spend & Transactions
    monthly_agg = act.groupby('MONTH').agg(
        total_spend=('TOTAL_SPEND_RECONSTRUCTED', 'sum'),
        avg_spend=('TOTAL_SPEND_RECONSTRUCTED', 'mean'),
        total_txns=('TRANSACTIONS', 'sum'),
        avg_txns=('TRANSACTIONS', 'mean'),
        active_custs=('CUSTOMER_ID', 'nunique')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    color1, color2 = '#1f77b4', '#2ca02c'
    ax1.plot(monthly_agg['MONTH'], monthly_agg['total_spend'] / 1000, marker='o', color=color1, linewidth=2.2, label='Total Monthly Spend ($K)')
    ax1.set_xlabel('Month', fontsize=11)
    ax1.set_ylabel('Total Spend ($ in Thousands)', color=color1, fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(monthly_agg['MONTH'], monthly_agg['total_txns'], marker='s', color=color2, linewidth=2.2, linestyle='--', label='Total Transactions')
    ax2.set_ylabel('Total Transactions', color=color2, fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)

    cutoff_date = pd.to_datetime('2024-03-31')
    ax1.axvline(cutoff_date, color='#d62728', linestyle=':', linewidth=2.5, label='Prediction Cutoff (2024-03-31)')
    ax1.text(pd.to_datetime('2024-03-15'), ax1.get_ylim()[1]*0.88, 'Prediction Cutoff\n(March 31, 2024)', color='#d62728', ha='right', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#d62728'))
    ax1.axvspan(pd.to_datetime('2024-04-01'), pd.to_datetime('2024-06-01'), color='#ffcccc', alpha=0.3, label='Target Outcome Window (Apr-Jun 2024)')
    plt.title('Monthly Store Spend & Transactions Across 18 Months (Jan 2023 - Jun 2024)', fontsize=13, fontweight='bold', pad=15)
    save_fig('03_monthly_trends_overall.png')

    # Chart 4: Age Distribution
    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [0.3, 0.7]})
    sns.boxplot(data=df_cust, x='AGE', y='CHURNED', orient='h', palette=['#2b5c8f', '#d95f02'], ax=ax_box)
    ax_box.set_yticks([0, 1])
    ax_box.set_yticklabels(['Retained (0)', 'Churned (1)'])
    ax_box.set_ylabel('')
    ax_box.set_title('Member Age Distribution by Churn Status (78 Missing Ages Excluded)', fontsize=12, fontweight='bold', pad=10)

    sns.histplot(data=df_cust, x='AGE', hue='CHURNED', element='step', stat='density', common_norm=False, palette=['#2b5c8f', '#d95f02'], ax=ax_hist, bins=25)
    ax_hist.set_xlabel('Member Age (Years)', fontsize=11)
    ax_hist.set_ylabel('Density', fontsize=11)
    ax_hist.legend(title='Status', labels=['Churned (1)', 'Retained (0)'])
    save_fig('04_customer_age_distribution.png')

    # Chart 5: Categories
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    tier_order = ['Silver', 'Gold', 'Platinum']
    tier_churn = df_cust.groupby('MEMBERSHIP_TIER')['CHURNED'].agg(['count', 'mean']).reindex(tier_order).reset_index()
    tier_churn['rate'] = tier_churn['mean'] * 100
    sns.barplot(data=tier_churn, x='MEMBERSHIP_TIER', y='rate', ax=axes[0,0], palette=['#a6bddb', '#ece7f2', '#fff7bc'], edgecolor='black')
    for i, row in tier_churn.iterrows():
        axes[0,0].text(i, row['rate'] + 0.8, f"{row['rate']:.1f}%\n(N={row['count']})", ha='center', fontsize=9, fontweight='bold')
    axes[0,0].set_title('Churn Rate by Membership Tier', fontsize=11, fontweight='bold')
    axes[0,0].set_ylabel('Churn Rate (%)')
    axes[0,0].set_ylim(0, 45)

    price_col = 'HOME_STORE_PRICE_TIER_CLEANED' if 'HOME_STORE_PRICE_TIER_CLEANED' in df_cust.columns else 'HOME_STORE_PRICE_TIER'
    price_order = ['Value', 'Mainstream', 'Upscale', 'Unknown']
    price_churn = df_cust.groupby(price_col)['CHURNED'].agg(['count', 'mean']).reindex(price_order).reset_index()
    price_churn['rate'] = price_churn['mean'] * 100
    sns.barplot(data=price_churn, x=price_col, y='rate', ax=axes[0,1], palette='Blues_d', edgecolor='black')
    for i, row in price_churn.iterrows():
        axes[0,1].text(i, row['rate'] + 0.8, f"{row['rate']:.1f}%\n(N={row['count']})", ha='center', fontsize=9, fontweight='bold')
    axes[0,1].set_title('Churn Rate by Home Store Price Tier', fontsize=11, fontweight='bold')
    axes[0,1].set_ylabel('Churn Rate (%)')
    axes[0,1].set_ylim(0, 45)

    gender_col = 'GENDER_CLEANED' if 'GENDER_CLEANED' in df_cust.columns else 'GENDER'
    gender_order = ['F', 'M', 'Other', 'Unknown']
    gender_churn = df_cust.groupby(gender_col)['CHURNED'].agg(['count', 'mean']).reindex(gender_order).reset_index()
    gender_churn['rate'] = gender_churn['mean'] * 100
    sns.barplot(data=gender_churn, x=gender_col, y='rate', ax=axes[1,0], palette='Set2', edgecolor='black')
    for i, row in gender_churn.iterrows():
        axes[1,0].text(i, row['rate'] + 0.8, f"{row['rate']:.1f}%\n(N={row['count']})", ha='center', fontsize=9, fontweight='bold')
    axes[1,0].set_title('Churn Rate by Member Gender', fontsize=11, fontweight='bold')
    axes[1,0].set_ylabel('Churn Rate (%)')
    axes[1,0].set_ylim(0, 45)

    city_churn = df_cust.groupby('CITY')['CHURNED'].agg(['count', 'mean']).sort_values(by='mean', ascending=False).reset_index()
    city_churn['rate'] = city_churn['mean'] * 100
    sns.barplot(data=city_churn, y='CITY', x='rate', ax=axes[1,1], palette='viridis', edgecolor='black')
    for i, row in city_churn.iterrows():
        axes[1,1].text(row['rate'] + 0.5, i, f"{row['rate']:.1f}%", ha='left', va='center', fontsize=8, fontweight='bold')
    axes[1,1].set_title('Churn Rate across 10 Cities', fontsize=11, fontweight='bold')
    axes[1,1].set_xlabel('Churn Rate (%)')
    axes[1,1].set_xlim(0, 45)
    save_fig('05_churn_by_categories.png')

    # Chart 6: Pre-cutoff behavior
    act_pre = act[act['MONTH'] < '2024-04-01'].copy()
    cust_pre_summary = act_pre.groupby('CUSTOMER_ID').agg(
        total_spend_pre=('TOTAL_SPEND_RECONSTRUCTED', 'sum'),
        monthly_spend_mean=('TOTAL_SPEND_RECONSTRUCTED', 'mean'),
        monthly_txns_mean=('TRANSACTIONS', 'mean'),
        monthly_app_mean=('APP_SESSIONS', 'mean'),
        monthly_email_mean=('EMAILS_OPENED', 'mean'),
        total_coupons=('COUPONS_REDEEMED', 'sum'),
        total_tickets=('SUPPORT_TICKETS', 'sum'),
        complaint_logged=('COMPLAINT_FLAG', 'max')
    ).reset_index()

    df_comp = df_cust.merge(cust_pre_summary, on='CUSTOMER_ID', how='left')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=df_comp, x='CHURNED', y='monthly_spend_mean', palette=['#2b5c8f', '#d95f02'], ax=axes[0], showfliers=False)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Retained (0)', 'Churned (1)'])
    axes[0].set_title('Pre-Cutoff Mean Monthly Spend ($/mo)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Mean Monthly Spend ($)')

    sns.boxplot(data=df_comp, x='CHURNED', y='monthly_txns_mean', palette=['#2b5c8f', '#d95f02'], ax=axes[1], showfliers=False)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Retained (0)', 'Churned (1)'])
    axes[1].set_title('Pre-Cutoff Mean Monthly Transactions (Txns/mo)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Mean Monthly Transactions')
    save_fig('06_historical_spending_behavior.png')

    # Chart 7: Support & Engagement
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    complaint_by_churn = df_comp.groupby('CHURNED')['complaint_logged'].mean() * 100
    bars1 = axes[0].bar(['Retained (0)', 'Churned (1)'], complaint_by_churn, color=['#2b5c8f', '#d95f02'], width=0.5, edgecolor='black')
    for bar, val in zip(bars1, complaint_by_churn):
        axes[0].text(bar.get_x() + bar.get_width()/2.0, val + 0.8, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    axes[0].set_title('Members Logging >= 1 Formal Complaint (Pre-Cutoff)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Share with Formal Complaint (%)')
    axes[0].set_ylim(0, max(complaint_by_churn)*1.3)

    app_by_churn = df_comp.groupby('CHURNED')['monthly_app_mean'].mean()
    bars2 = axes[1].bar(['Retained (0)', 'Churned (1)'], app_by_churn, color=['#2b5c8f', '#d95f02'], width=0.5, edgecolor='black')
    for bar, val in zip(bars2, app_by_churn):
        axes[1].text(bar.get_x() + bar.get_width()/2.0, val + 0.1, f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    axes[1].set_title('Mean App Sessions per Month (Pre-Cutoff)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Avg Monthly App Sessions')
    axes[1].set_ylim(0, max(app_by_churn)*1.3)
    save_fig('07_engagement_and_friction_comparison.png')

    print(f'All 7 EDA charts generated successfully in {CHARTS_DIR}.')


if __name__ == '__main__':
    generate_eda_charts()
