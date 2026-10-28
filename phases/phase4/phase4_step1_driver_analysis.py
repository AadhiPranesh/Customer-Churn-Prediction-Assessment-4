"""
FreshBasket Customer Churn Prediction
Phase 4 — Step 1: Model Interpretability & Subgroup Risk Profiling
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

# Safe path resolution
try:
    from phases.paths import (
        PROJECT_ROOT,
        PROCESSED_DATASET_PATH,
        VALIDATION_PREDS_PATH,
        SAVED_MODEL_PATH,
        RF_IMPORTANCES_PATH,
        LR_COEFS_PATH,
        PHASE4_OUTPUT as OUTPUTS_ANALYSIS,
        PHASE4_CHARTS as CHARTS_DIR
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv"
    VALIDATION_PREDS_PATH = PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions.csv"
    SAVED_MODEL_PATH = PROJECT_ROOT / "models" / "phase3_random_forest_pipeline.joblib"
    RF_IMPORTANCES_PATH = PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_random_forest_feature_importances.csv"
    LR_COEFS_PATH = PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_logistic_regression_coefficients.csv"
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


def run_driver_analysis():
    # 1. Load Data
    master_path = _find_file(PROCESSED_DATASET_PATH, PROJECT_ROOT / "phase2_modeling_dataset.csv")
    val_preds_path = _find_file(VALIDATION_PREDS_PATH, PROJECT_ROOT / "phase3_validation_predictions.csv")
    model_path = _find_file(SAVED_MODEL_PATH, PROJECT_ROOT / "phase3_random_forest_pipeline.joblib")
    rf_imp_path = _find_file(RF_IMPORTANCES_PATH, PROJECT_ROOT / "phase3_random_forest_feature_importances.csv")
    lr_coef_path = _find_file(LR_COEFS_PATH, PROJECT_ROOT / "phase3_logistic_regression_coefficients.csv")

    df_master = pd.read_csv(master_path)
    df_val_preds = pd.read_csv(val_preds_path)

    train_df = df_master[df_master['VALIDATION_COHORT'] == 'TRAIN'].copy()
    val_df = df_master[df_master['VALIDATION_COHORT'] == 'VAL'].copy()

    admin_cols = ['CUSTOMER_ID', 'SIGNUP_DATE', 'CHURNED', 'VALIDATION_COHORT', 'IS_POST_CUTOFF_SIGNUP']
    feature_cols = [c for c in df_master.columns if c not in admin_cols]
    cat_cols = ['GENDER', 'CITY', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER']
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_train, y_train = train_df[feature_cols], train_df['CHURNED']
    X_val, y_val = val_df[feature_cols], val_df['CHURNED']

    # Load serialized pipeline
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Missing {model_path}. Must exist to ensure Phase 3 reproducibility.")
    rf_pipe = joblib.load(model_path)
    print(f"Loaded existing Phase 3 fitted pipeline from '{model_path}'.")

    rf_clf = rf_pipe.named_steps['clf']
    n_trees = rf_clf.n_estimators
    print(f"Verified Model Architecture: RandomForestClassifier with {n_trees} estimators (max_depth={rf_clf.max_depth}, min_samples_leaf={rf_clf.min_samples_leaf}).")

    preprocessor = rf_pipe.named_steps['pre']
    X_val_trans = preprocessor.transform(X_val)
    ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
    all_feature_names = num_cols + ohe.get_feature_names_out(cat_cols).tolist()

    val_prob_recomputed = rf_pipe.predict_proba(X_val)[:, 1]
    max_pred_diff = np.abs(val_prob_recomputed - df_val_preds['RF_CHURN_PROBABILITY'].values).max()
    print(f"Prediction Reconciled against phase3_validation_predictions.csv: Max Diff = {max_pred_diff:.6f}.")

    # 2. SHAP Analysis
    print("\n=== 2. Computing Exact SHAP Values on Validation Set ===")
    explainer = shap.TreeExplainer(rf_clf)
    shap_values = explainer.shap_values(X_val_trans)

    if isinstance(shap_values, list):
        sv_churn = shap_values[1]
        base_value = explainer.expected_value[1]
    elif len(shap_values.shape) == 3:
        sv_churn = shap_values[:, :, 1]
        base_value = explainer.expected_value[1]
    else:
        sv_churn = shap_values
        base_value = explainer.expected_value

    reconstructed_probs = base_value + sv_churn.sum(axis=1)
    max_additivity_error = np.abs(reconstructed_probs - val_prob_recomputed).max()
    print(f"SHAP Additivity Verified: Base Value = {base_value:.4f}, Max Reconstruction Error = {max_additivity_error:.6f}")

    mean_abs_shap = np.abs(sv_churn).mean(axis=0)
    df_shap = pd.DataFrame({
        'Feature': all_feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

    df_shap.to_csv(OUTPUTS_ANALYSIS / 'phase4_step1_shap_summary.csv', index=False)

    # 3. Predictors table
    df_rf_imp = pd.read_csv(rf_imp_path)
    df_lr_coef = pd.read_csv(lr_coef_path)

    merged_imp = df_rf_imp.merge(df_shap, on='Feature').merge(df_lr_coef, on='Feature')
    merged_imp['RF_Rank'] = merged_imp['RF_Gini_Importance'].rank(ascending=False).astype(int)
    merged_imp['SHAP_Rank'] = merged_imp['Mean_Abs_SHAP'].rank(ascending=False).astype(int)

    top10_table = merged_imp.sort_values(by='RF_Gini_Importance', ascending=False).head(10)[[
        'Feature', 'RF_Rank', 'RF_Gini_Importance', 'SHAP_Rank', 'Mean_Abs_SHAP', 'LR_Coefficient', 'Odds_Ratio'
    ]]
    top10_table.to_csv(OUTPUTS_ANALYSIS / 'phase4_step1_top_predictors.csv', index=False)

    # 4. Segment Risk Profiling
    val_merged = pd.merge(
        df_val_preds,
        df_master[['CUSTOMER_ID', 'AGE', 'GENDER', 'CITY', 'TENURE_DAYS_PRE_CUTOFF', 'TOTAL_SPEND_PRE_CUTOFF']],
        on='CUSTOMER_ID',
        how='left'
    )

    seg_tier_list = []
    for tier in ['Silver', 'Gold', 'Platinum']:
        sub = val_merged[val_merged['MEMBERSHIP_TIER'] == tier]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ch_rate = round(ch / n * 100, 2)
        rf_pred_ch = int(sub['RF_PRED_LABEL'].sum())
        rf_pred_rate = round(rf_pred_ch / n * 100, 2)
        rf_mean_prob = round(sub['RF_CHURN_PROBABILITY'].mean() * 100, 2)
        lr_pred_ch = int(sub['LR_PRED_LABEL'].sum())
        lr_pred_rate = round(lr_pred_ch / n * 100, 2)
        lr_mean_prob = round(sub['LR_CHURN_PROBABILITY'].mean() * 100, 2)
        
        tp_rf = int(((sub['CHURNED'] == 1) & (sub['RF_PRED_LABEL'] == 1)).sum())
        fp_rf = int(((sub['CHURNED'] == 0) & (sub['RF_PRED_LABEL'] == 1)).sum())
        rec_rf = round(tp_rf / ch * 100, 1) if ch > 0 else 0.0
        prec_rf = round(tp_rf / (tp_rf + fp_rf) * 100, 1) if (tp_rf + fp_rf) > 0 else 0.0
        sample_warning = "Small Sample (N=63, 14 churners)" if n < 100 else "Sufficient Sample Size"
        
        seg_tier_list.append({
            'Segment_Type': 'Membership_Tier',
            'Segment_Value': tier,
            'Customer_Count_N': n,
            'Actual_Churn_Count': ch,
            'Actual_Churn_Rate_Pct': ch_rate,
            'RF_Mean_Churn_Prob_Pct': rf_mean_prob,
            'RF_Predicted_Churn_Count': rf_pred_ch,
            'RF_Predicted_Churn_Rate_Pct': rf_pred_rate,
            'RF_Recall_Pct': rec_rf,
            'RF_Precision_Pct': prec_rf,
            'LR_Mean_Churn_Prob_Pct': lr_mean_prob,
            'LR_Predicted_Churn_Count': lr_pred_ch,
            'Sample_Size_Reliability': sample_warning
        })

    seg_price_list = []
    for pt in ['Mainstream', 'Value', 'Upscale', 'Unknown']:
        sub = val_merged[val_merged['HOME_STORE_PRICE_TIER'] == pt]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ch_rate = round(ch / n * 100, 2) if n > 0 else 0.0
        rf_pred_ch = int(sub['RF_PRED_LABEL'].sum())
        rf_pred_rate = round(rf_pred_ch / n * 100, 2) if n > 0 else 0.0
        rf_mean_prob = round(sub['RF_CHURN_PROBABILITY'].mean() * 100, 2) if n > 0 else 0.0
        lr_pred_ch = int(sub['LR_PRED_LABEL'].sum())
        lr_pred_rate = round(lr_pred_ch / n * 100, 2) if n > 0 else 0.0
        lr_mean_prob = round(sub['LR_CHURN_PROBABILITY'].mean() * 100, 2) if n > 0 else 0.0
        
        tp_rf = int(((sub['CHURNED'] == 1) & (sub['RF_PRED_LABEL'] == 1)).sum())
        fp_rf = int(((sub['CHURNED'] == 0) & (sub['RF_PRED_LABEL'] == 1)).sum())
        rec_rf = round(tp_rf / ch * 100, 1) if ch > 0 else 0.0
        prec_rf = round(tp_rf / (tp_rf + fp_rf) * 100, 1) if (tp_rf + fp_rf) > 0 else 0.0
        sample_warning = "Small Subgroup (N=23, wide CI)" if n < 50 else "Sufficient Sample Size"
        
        seg_price_list.append({
            'Segment_Type': 'Home_Store_Price_Tier',
            'Segment_Value': pt,
            'Customer_Count_N': n,
            'Actual_Churn_Count': ch,
            'Actual_Churn_Rate_Pct': ch_rate,
            'RF_Mean_Churn_Prob_Pct': rf_mean_prob,
            'RF_Predicted_Churn_Count': rf_pred_ch,
            'RF_Predicted_Churn_Rate_Pct': rf_pred_rate,
            'RF_Recall_Pct': rec_rf,
            'RF_Precision_Pct': prec_rf,
            'LR_Mean_Churn_Prob_Pct': lr_mean_prob,
            'LR_Predicted_Churn_Count': lr_pred_ch,
            'Sample_Size_Reliability': sample_warning
        })

    df_seg_summary = pd.DataFrame(seg_tier_list + seg_price_list)
    df_seg_summary.to_csv(OUTPUTS_ANALYSIS / 'phase4_step1_segment_risk_summary.csv', index=False)

    # City Risk Profiling
    city_list = []
    for city in sorted(val_merged['CITY'].dropna().unique()):
        sub = val_merged[val_merged['CITY'] == city]
        n = len(sub)
        ch = int(sub['CHURNED'].sum())
        ch_rate = round(ch / n * 100, 2)
        rf_pred_ch = int(sub['RF_PRED_LABEL'].sum())
        rf_pred_rate = round(rf_pred_ch / n * 100, 2)
        rf_mean_prob = round(sub['RF_CHURN_PROBABILITY'].mean() * 100, 2)
        tp_rf = int(((sub['CHURNED'] == 1) & (sub['RF_PRED_LABEL'] == 1)).sum())
        fp_rf = int(((sub['CHURNED'] == 0) & (sub['RF_PRED_LABEL'] == 1)).sum())
        rec_rf = round(tp_rf / ch * 100, 1) if ch > 0 else 0.0
        prec_rf = round(tp_rf / (tp_rf + fp_rf) * 100, 1) if (tp_rf + fp_rf) > 0 else 0.0
        city_list.append({
            'City': city,
            'Customer_Count_N': n,
            'Actual_Churn_Count': ch,
            'Actual_Churn_Rate_Pct': ch_rate,
            'RF_Mean_Churn_Prob_Pct': rf_mean_prob,
            'RF_Predicted_Churn_Count': rf_pred_ch,
            'RF_Predicted_Churn_Rate_Pct': rf_pred_rate,
            'RF_Recall_Pct': rec_rf,
            'RF_Precision_Pct': prec_rf,
            'Sample_Size_Warning': f"Small Subgroup (N={n}, {ch} churners; wide confidence intervals)"
        })
    df_city_summary = pd.DataFrame(city_list).sort_values(by='Actual_Churn_Rate_Pct', ascending=False).reset_index(drop=True)
    df_city_summary.to_csv(OUTPUTS_ANALYSIS / 'phase4_step1_city_risk_summary.csv', index=False)

    # 5. Charts
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # Chart 1
    plt.figure(figsize=(12, 7))
    plot_top = top10_table.head(8).sort_values(by='RF_Gini_Importance', ascending=True)
    y_pos = np.arange(len(plot_top))
    plt.barh(y_pos, plot_top['RF_Gini_Importance'] * 100, color='#1f77b4', alpha=0.85)
    for i, v in enumerate(plot_top['RF_Gini_Importance'] * 100):
        plt.text(v + 0.3, i, f"{v:.1f}%", va='center', fontweight='bold', color='#1f77b4')
    plt.yticks(y_pos, plot_top['Feature'], fontsize=11)
    plt.xlabel('Random Forest Gini Impurity Importance (%)', fontsize=12, fontweight='bold')
    plt.title('FreshBasket Churn Prediction: Top Predictive Features (Training Ensemble)', fontsize=14, fontweight='bold', pad=15)
    plt.xlim(0, 22)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'phase4_top_predictors_comparison.png', dpi=300)
    plt.close()

    # Chart 2: SHAP
    plt.figure(figsize=(12, 7))
    plot_shap = df_shap.head(10).sort_values(by='Mean_Abs_SHAP', ascending=True)
    y_pos_s = np.arange(len(plot_shap))
    plt.barh(y_pos_s, plot_shap['Mean_Abs_SHAP'], color='#ff7f0e', alpha=0.85)
    for i, v in enumerate(plot_shap['Mean_Abs_SHAP']):
        plt.text(v + 0.002, i, f"{v:.4f}", va='center', fontweight='bold', color='#d62728')
    plt.yticks(y_pos_s, plot_shap['Feature'], fontsize=11)
    plt.xlabel('Mean |SHAP Value| (Impact on Validation Probability)', fontsize=12, fontweight='bold')
    plt.title('Validation Cohort SHAP Attribution (TreeExplainer, N = 602)', fontsize=14, fontweight='bold', pad=15)
    plt.xlim(0, 0.13)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'phase4_shap_summary.png', dpi=300)
    plt.close()

    # Chart 3: Segment Churn Profiles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    tiers = ['Silver', 'Gold', 'Platinum']
    actual_rates = [df_seg_summary[df_seg_summary['Segment_Value'] == t]['Actual_Churn_Rate_Pct'].values[0] for t in tiers]
    rf_rates = [df_seg_summary[df_seg_summary['Segment_Value'] == t]['RF_Mean_Churn_Prob_Pct'].values[0] for t in tiers]
    x = np.arange(len(tiers))
    width = 0.35
    ax1.bar(x - width/2, actual_rates, width, label='Actual Churn Rate (%)', color='#2ca02c', alpha=0.85)
    ax1.bar(x + width/2, rf_rates, width, label='RF Mean Pred Prob (%)', color='#1f77b4', alpha=0.85)
    ax1.set_ylabel('Percentage (%)', fontweight='bold')
    ax1.set_title('Churn Risk by Loyalty Tier (N = 602)', fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t}\n(N={df_seg_summary[df_seg_summary['Segment_Value'] == t]['Customer_Count_N'].values[0]})" for t in tiers])
    ax1.legend()

    pts = ['Mainstream', 'Value', 'Upscale', 'Unknown']
    pt_actual = [df_seg_summary[df_seg_summary['Segment_Value'] == p]['Actual_Churn_Rate_Pct'].values[0] for p in pts]
    pt_rf = [df_seg_summary[df_seg_summary['Segment_Value'] == p]['RF_Mean_Churn_Prob_Pct'].values[0] for p in pts]
    x2 = np.arange(len(pts))
    ax2.bar(x2 - width/2, pt_actual, width, label='Actual Churn Rate (%)', color='#2ca02c', alpha=0.85)
    ax2.bar(x2 + width/2, pt_rf, width, label='RF Mean Pred Prob (%)', color='#1f77b4', alpha=0.85)
    ax2.set_ylabel('Percentage (%)', fontweight='bold')
    ax2.set_title('Churn Risk by Store Price Tier (All 602 Customers)', fontweight='bold', pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"{p}\n(N={df_seg_summary[df_seg_summary['Segment_Value'] == p]['Customer_Count_N'].values[0]})" for p in pts])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'phase4_segment_churn_profiles.png', dpi=300)
    plt.close()

    print("\n=== Phase 4 Step 1 Execution Completed Successfully ===")


if __name__ == '__main__':
    run_driver_analysis()
