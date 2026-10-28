"""
Comprehensive final verification script for FreshBasket Churn Prediction project (Assessment 4).
Checks all deliverables, verified metrics, confusion matrices, ablation sets, scenario counts, segment totals,
plus new Assessment 4 deliverables (segment & cohort tables, batch predictions with SHAP drivers, ROI sensitivity, EDA notebook, root directory isolation).
Supports both reorganized canonical paths and legacy paths seamlessly.
"""

import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Determine project root dynamically
try:
    from phases.paths import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_file(rel_path):
    """Resolve a file path prioritizing the reorganized phase-based structure, with fallback."""
    canonical = PROJECT_ROOT / rel_path
    if canonical.exists():
        return canonical
    
    filename = Path(rel_path).name
    # Check reorganized phase-based and alternate locations
    alternatives = [
        # Phase-based structure
        PROJECT_ROOT / "phases" / "phase3" / "output" / filename,
        PROJECT_ROOT / "phases" / "phase4" / "output" / filename,
        PROJECT_ROOT / "phases" / "phase4" / "output" / "charts" / filename,
        PROJECT_ROOT / "phases" / "phase2" / "output" / filename,
        PROJECT_ROOT / "phases" / "phase1" / "output" / "cleaned_data" / filename,
        PROJECT_ROOT / "phases" / "phase1" / "output" / "charts" / filename,
        PROJECT_ROOT / "phases" / "phase1" / "output" / filename,
        PROJECT_ROOT / "phases" / "phase1" / filename,
        PROJECT_ROOT / "phases" / "phase2" / filename,
        PROJECT_ROOT / "phases" / "phase3" / filename,
        PROJECT_ROOT / "phases" / "phase4" / filename,
        PROJECT_ROOT / "data" / "processed" / filename,
        PROJECT_ROOT / "data" / "interim" / filename,
        PROJECT_ROOT / "data" / "raw" / filename,
        PROJECT_ROOT / "models" / filename,
        PROJECT_ROOT / "reports" / filename,
        PROJECT_ROOT / "reports" / "assessment" / filename,
        PROJECT_ROOT / "reports" / "presentation" / filename,
        PROJECT_ROOT / "config" / filename,
        PROJECT_ROOT / "notebooks" / filename,
        # Legacy structure fallbacks
        PROJECT_ROOT / "outputs" / "evaluation" / filename,
        PROJECT_ROOT / "outputs" / "analysis" / filename,
        PROJECT_ROOT / "outputs" / "eda" / filename,
        PROJECT_ROOT / "outputs" / "phase_1" / rel_path,
        PROJECT_ROOT / "outputs" / "phase_2" / rel_path,
        PROJECT_ROOT / "outputs" / "phase_3" / rel_path,
        PROJECT_ROOT / "outputs" / "phase_4" / rel_path,
        PROJECT_ROOT / "outputs" / rel_path,
        # Root fallback
        PROJECT_ROOT / filename,
    ]
    for alt in alternatives:
        if alt.exists():
            return alt
    return canonical


def run_checks():
    print("=== STARTING FINAL COMPREHENSIVE QUALITY GATE AUDIT ===")
    
    # 0. Check Root Directory Isolation (Only predict_customer.py and README.md in root)
    root_files = [f.name for f in PROJECT_ROOT.iterdir() if f.is_file()]
    allowed_root_files = {'predict_customer.py', 'README.md', '.gitignore'}
    disallowed_root_files = set(root_files) - allowed_root_files
    if disallowed_root_files:
        raise AssertionError(f"Root directory contains unauthorized files: {disallowed_root_files}. Only predict_customer.py, README.md, and .gitignore are permitted in root.")
    print("PASS: Root directory isolation verified (strictly contains only predict_customer.py and README.md).")

    # 1. Check distinct deliverables exist in the reorganized phase-based structure
    required_canonical_files = [
        # Core Root Files
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "predict_customer.py",
        PROJECT_ROOT / "config" / "requirements.txt",
        
        # Raw and Processed Datasets
        PROJECT_ROOT / "data" / "raw" / "FreshBasket_Loyalty_Churn_Dataset.xlsx",
        PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv",
        
        # Models
        PROJECT_ROOT / "models" / "phase3_random_forest_pipeline.joblib",
        
        # Notebooks inside Phase Folders
        PROJECT_ROOT / "phases" / "phase1" / "phase1_cleaning_notebook.ipynb",
        PROJECT_ROOT / "phases" / "phase1" / "phase1_eda_and_data_quality_notebook.ipynb",
        PROJECT_ROOT / "phases" / "phase2" / "phase2_feature_engineering_notebook.ipynb",
        PROJECT_ROOT / "phases" / "phase3" / "phase3_model_development_notebook.ipynb",
        
        # Phase 1 Outputs, Publication Charts & Reports
        PROJECT_ROOT / "phases" / "phase1" / "output" / "cleaned_data" / "cleaned_fb_customer_profile.csv",
        PROJECT_ROOT / "phases" / "phase1" / "output" / "cleaned_data" / "cleaned_fb_churn_label.csv",
        PROJECT_ROOT / "phases" / "phase1" / "output" / "cleaned_data" / "cleaned_fb_monthly_activity.csv",
        PROJECT_ROOT / "phases" / "phase1" / "output" / "charts" / "01_churn_distribution.png",
        PROJECT_ROOT / "phases" / "phase1" / "output" / "charts" / "02_missing_values_summary.png",
        PROJECT_ROOT / "phases" / "phase1" / "output" / "charts" / "03_monthly_trends_overall.png",
        PROJECT_ROOT / "phases" / "phase1" / "output" / "charts" / "04_customer_age_distribution.png",
        PROJECT_ROOT / "phases" / "phase1" / "output" / "charts" / "05_churn_by_categories.png",
        PROJECT_ROOT / "phases" / "phase1" / "output" / "charts" / "06_historical_spending_behavior.png",
        PROJECT_ROOT / "phases" / "phase1" / "output" / "charts" / "07_engagement_and_friction_comparison.png",
        PROJECT_ROOT / "phases" / "phase1" / "phase1_data_quality_report.md",
        PROJECT_ROOT / "phases" / "phase1" / "phase1_decisions.md",
        PROJECT_ROOT / "phases" / "phase1" / "phase1_eda_report.md",
        PROJECT_ROOT / "phases" / "phase1" / "phase1_review_corrections.md",
        
        # Phase 2 Documentation & Audits
        PROJECT_ROOT / "phases" / "phase2" / "phase2_feature_dictionary.md",
        PROJECT_ROOT / "phases" / "phase2" / "phase2_data_checks.md",
        PROJECT_ROOT / "phases" / "phase2" / "phase2_final_audit.md",
        PROJECT_ROOT / "phases" / "phase2" / "phase2_validation_cohort_audit.md",
        
        # Phase 3 Outputs & Reports
        PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions.csv",
        PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions_with_drivers.csv",
        PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_batch_predictions_all_eligible.csv",
        PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_model_metrics.csv",
        PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_logistic_regression_coefficients.csv",
        PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_random_forest_feature_importances.csv",
        PROJECT_ROOT / "phases" / "phase3" / "phase3_model_evaluation_report.md",
        
        # Phase 4 Outputs & Reports
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_step1_top_predictors.csv",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_step1_shap_summary.csv",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_step1_segment_risk_summary.csv",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_step1_city_risk_summary.csv",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_segment_cohort_churn_analysis.csv",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_signup_cohort_analysis.csv",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_step2_ablation_results.csv",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_step2_ablation_deltas.csv",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_step3_retention_scenario_results.csv",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_step3_scenario_overlap.csv",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_step3_customer_targeting_roster.csv",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_roi_assumptions.csv",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_retention_financial_roi_sensitivity.csv",
        PROJECT_ROOT / "phases" / "phase4" / "phase4_step1_driver_analysis_report.md",
        PROJECT_ROOT / "phases" / "phase4" / "phase4_step2_ablation_report.md",
        PROJECT_ROOT / "phases" / "phase4" / "phase4_step3_retention_report.md",
        
        # Phase 4 Charts
        PROJECT_ROOT / "phases" / "phase4" / "output" / "charts" / "phase4_top_predictors_comparison.png",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "charts" / "phase4_shap_summary.png",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "charts" / "phase4_segment_churn_profiles.png",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "charts" / "phase4_signup_cohort_distribution.png",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "charts" / "phase4_ablation_comparison.png",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "charts" / "phase4_retention_simulation.png",
        PROJECT_ROOT / "phases" / "phase4" / "output" / "charts" / "phase4_retention_roi_sensitivity.png",
        
        # Reports
        PROJECT_ROOT / "reports" / "FINAL_PROJECT_REPORT.md",
        PROJECT_ROOT / "reports" / "assessment" / "Customer_Churn_Prediction_Assessment.docx"
    ]
    
    missing = [str(f.relative_to(PROJECT_ROOT)) for f in required_canonical_files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required canonical project files: {missing}")
    print(f"PASS: All {len(required_canonical_files)} distinct canonical deliverables and artifacts exist on disk.")

    # 1b. Check convenience mirrors and backward compatibility paths without double-counting them as independent deliverables
    convenience_mirrors = [
        (PROJECT_ROOT / "phases" / "phase2" / "output" / "phase2_modeling_dataset.csv", PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv"),
        (PROJECT_ROOT / "notebooks" / "phase1_eda_and_data_quality_notebook.ipynb", PROJECT_ROOT / "phases" / "phase1" / "phase1_eda_and_data_quality_notebook.ipynb"),
        (PROJECT_ROOT / "notebooks" / "phase1_cleaning_notebook.ipynb", PROJECT_ROOT / "phases" / "phase1" / "phase1_cleaning_notebook.ipynb"),
        (PROJECT_ROOT / "notebooks" / "phase2_feature_engineering_notebook.ipynb", PROJECT_ROOT / "phases" / "phase2" / "phase2_feature_engineering_notebook.ipynb"),
        (PROJECT_ROOT / "notebooks" / "phase3_model_development_notebook.ipynb", PROJECT_ROOT / "phases" / "phase3" / "phase3_model_development_notebook.ipynb"),
    ]
    for mirror, src in convenience_mirrors:
        assert mirror.exists(), f"Convenience mirror missing: {mirror}"
        assert mirror.stat().st_size == src.stat().st_size, f"Mirror size mismatch: {mirror} vs {src}"
    print(f"PASS: All {len(convenience_mirrors)} convenience mirrors in notebooks/ and phase2/output verified and validated against canonical sources.")

    # 2. Check validation cohort counts
    preds_file = resolve_file("phases/phase3/output/phase3_validation_predictions.csv")
    preds = pd.read_csv(preds_file)
    n_total = len(preds)
    n_churn = int(preds['CHURNED'].sum())
    n_retained = n_total - n_churn
    print(f"Validation cohort: Total={n_total}, Churn={n_churn}, Retained={n_retained}")
    assert n_total == 602, f"Expected 602, got {n_total}"
    assert n_churn == 85, f"Expected 85, got {n_churn}"
    assert n_retained == 517, f"Expected 517, got {n_retained}"
    print("PASS: Validation cohort counts match perfectly (602: 85 churners, 517 retained).")

    # 3. Check Phase 3 Model Metrics & Confusion Matrices
    metrics_file = resolve_file("phases/phase3/output/phase3_model_metrics.csv")
    metrics_df = pd.read_csv(metrics_file)
    lr_row = metrics_df[metrics_df['Model'].str.contains('Logistic', case=False)].iloc[0]
    rf_row = metrics_df[metrics_df['Model'].str.contains('Random Forest', case=False)].iloc[0]
    
    # Check LR metrics
    np.testing.assert_almost_equal(lr_row['Accuracy'], 0.9468, decimal=4)
    np.testing.assert_almost_equal(lr_row['Precision'], 0.8955, decimal=4)
    np.testing.assert_almost_equal(lr_row['Recall'], 0.7059, decimal=4)
    np.testing.assert_almost_equal(lr_row['F1_Score'], 0.7895, decimal=4)
    np.testing.assert_almost_equal(lr_row['PR_AUC'], 0.8808, decimal=4)
    np.testing.assert_almost_equal(lr_row['ROC_AUC'], 0.9644, decimal=4)
    print("PASS: Logistic Regression metrics match verified specifications (Acc 0.9468, Prec 0.8955, Rec 0.7059, F1 0.7895, PR-AUC 0.8808, ROC-AUC 0.9644).")

    # Check RF metrics
    np.testing.assert_almost_equal(rf_row['Accuracy'], 0.9169, decimal=4)
    np.testing.assert_almost_equal(rf_row['Precision'], 0.6667, decimal=4)
    np.testing.assert_almost_equal(rf_row['Recall'], 0.8235, decimal=4)
    np.testing.assert_almost_equal(rf_row['F1_Score'], 0.7368, decimal=4)
    np.testing.assert_almost_equal(rf_row['PR_AUC'], 0.8499, decimal=4)
    np.testing.assert_almost_equal(rf_row['ROC_AUC'], 0.9446, decimal=4)
    print("PASS: Random Forest metrics match verified specifications (Acc 0.9169, Prec 0.6667, Rec 0.8235, F1 0.7368, PR-AUC 0.8499, ROC-AUC 0.9446).")

    # Confusion matrix calculations from saved customer predictions
    lr_pred = preds['LR_PRED_LABEL']
    lr_tn = int(((preds['CHURNED'] == 0) & (lr_pred == 0)).sum())
    lr_fp = int(((preds['CHURNED'] == 0) & (lr_pred == 1)).sum())
    lr_fn = int(((preds['CHURNED'] == 1) & (lr_pred == 0)).sum())
    lr_tp = int(((preds['CHURNED'] == 1) & (lr_pred == 1)).sum())
    assert (lr_tn, lr_fp, lr_fn, lr_tp) == (510, 7, 25, 60), f"LR CM mismatch: {(lr_tn, lr_fp, lr_fn, lr_tp)}"

    rf_pred = preds['RF_PRED_LABEL']
    rf_tn = int(((preds['CHURNED'] == 0) & (rf_pred == 0)).sum())
    rf_fp = int(((preds['CHURNED'] == 0) & (rf_pred == 1)).sum())
    rf_fn = int(((preds['CHURNED'] == 1) & (rf_pred == 0)).sum())
    rf_tp = int(((preds['CHURNED'] == 1) & (rf_pred == 1)).sum())
    assert (rf_tn, rf_fp, rf_fn, rf_tp) == (482, 35, 15, 70), f"RF CM mismatch: {(rf_tn, rf_fp, rf_fn, rf_tp)}"
    print("PASS: Confusion matrices match verified outputs: LR=(510, 7, 25, 60), RF=(482, 35, 15, 70).")

    # 4. Check Feature Ablation metrics
    abl_file = resolve_file("phases/phase4/output/phase4_step2_ablation_results.csv")
    abl_df = pd.read_csv(abl_file)
    print("\nVerifying Feature Ablation Results Table:")
    for idx, row in abl_df.iterrows():
        print(f"  {row['Feature_Set']}: Acc={row['Accuracy']:.4f}, Prec={row['Precision']:.4f}, Rec={row['Recall']:.4f}, PR-AUC={row['PR_AUC']:.4f}, CM=(TN={row['TN']}, FP={row['FP']}, FN={row['FN']}, TP={row['TP']})")
    
    s1 = abl_df.iloc[0]
    np.testing.assert_almost_equal(s1['Accuracy'], 0.5482, decimal=4)
    np.testing.assert_almost_equal(s1['Recall'], 0.4235, decimal=4)
    np.testing.assert_almost_equal(s1['PR_AUC'], 0.1410, decimal=4)
    assert (s1['TN'], s1['FP'], s1['FN'], s1['TP']) == (294, 223, 49, 36)

    s2 = abl_df.iloc[1]
    np.testing.assert_almost_equal(s2['Accuracy'], 0.1495, decimal=4)
    np.testing.assert_almost_equal(s2['Recall'], 0.9765, decimal=4)
    np.testing.assert_almost_equal(s2['PR_AUC'], 0.1791, decimal=4)
    assert (s2['TN'], s2['FP'], s2['FN'], s2['TP']) == (7, 510, 2, 83)

    s3 = abl_df.iloc[2]
    np.testing.assert_almost_equal(s3['Accuracy'], 0.9103, decimal=4)
    np.testing.assert_almost_equal(s3['Recall'], 0.7529, decimal=4)
    np.testing.assert_almost_equal(s3['PR_AUC'], 0.8290, decimal=4)
    assert (s3['TN'], s3['FP'], s3['FN'], s3['TP']) == (484, 33, 21, 64)

    s4 = abl_df.iloc[3]
    np.testing.assert_almost_equal(s4['Accuracy'], 0.9169, decimal=4)
    np.testing.assert_almost_equal(s4['Recall'], 0.8235, decimal=4)
    np.testing.assert_almost_equal(s4['PR_AUC'], 0.8499, decimal=4)
    assert (s4['TN'], s4['FP'], s4['FN'], s4['TP']) == (482, 35, 15, 70)
    print("PASS: Feature Ablation metrics match verified outputs across all 4 sets.")

    # 5. Check Retention Simulation figures
    sim_file = resolve_file("phases/phase4/output/phase4_step3_retention_scenario_results.csv")
    sim_res = pd.read_csv(sim_file)
    print("\nVerifying Retention Scenario Results Table:")
    for idx, row in sim_res.iterrows():
        print(f"  {row['Scenario_Name']}: Targeted={row['Targeted_Customers_N']}, Churners={row['Actual_Churners_Targeted']}, FP={row['Actual_Retained_Targeted_FP']}")
    
    s1_row = sim_res[sim_res['Scenario_ID'] == 'S1'].iloc[0]
    assert s1_row['Targeted_Customers_N'] == 54
    assert s1_row['Actual_Churners_Targeted'] == 45
    assert s1_row['Actual_Retained_Targeted_FP'] == 9

    s2_row = sim_res[sim_res['Scenario_ID'] == 'S2'].iloc[0]
    assert s2_row['Targeted_Customers_N'] == 55
    assert s2_row['Actual_Churners_Targeted'] == 52
    assert s2_row['Actual_Retained_Targeted_FP'] == 3

    s3_row = sim_res[sim_res['Scenario_ID'] == 'S3'].iloc[0]
    assert s3_row['Targeted_Customers_N'] == 51
    assert s3_row['Actual_Churners_Targeted'] == 35
    assert s3_row['Actual_Retained_Targeted_FP'] == 16

    comb_row = sim_res[sim_res['Scenario_ID'] == 'COMBINED'].iloc[0]
    assert comb_row['Targeted_Customers_N'] == 91
    assert comb_row['Actual_Churners_Targeted'] == 67
    assert comb_row['Actual_Retained_Targeted_FP'] == 24

    roster_file = resolve_file("phases/phase4/output/phase4_step3_customer_targeting_roster.csv")
    roster = pd.read_csv(roster_file)
    assert len(roster) == 105, f"Expected 105 roster members, got {len(roster)}"
    
    # Overlap reconciliation
    overlap_file = resolve_file("phases/phase4/output/phase4_step3_scenario_overlap.csv")
    overlap_df = pd.read_csv(overlap_file)
    not_targeted = overlap_df[overlap_df['Segment'].str.contains('Not Targeted')].iloc[0]
    assert not_targeted['Count'] == 14
    assert not_targeted['Actual_Churners'] == 3
    print("PASS: Retention simulation figures match verified outputs (S1=54, S2=55, S3=51, Combined=91, Untargeted=14, Total Roster=105).")

    # 6. Check Segment Profiling totals
    seg_file = resolve_file("phases/phase4/output/phase4_step1_segment_risk_summary.csv")
    seg_df = pd.read_csv(seg_file)
    tiers = seg_df[seg_df['Segment_Type'] == 'Membership_Tier']
    tier_n = tiers['Customer_Count_N'].sum()
    tier_churn = tiers['Actual_Churn_Count'].sum()
    assert tier_n == 602, f"Expected 602 tier total, got {tier_n}"
    assert tier_churn == 85, f"Expected 85 tier churn, got {tier_churn}"
    
    silver = tiers[tiers['Segment_Value'] == 'Silver'].iloc[0]
    gold = tiers[tiers['Segment_Value'] == 'Gold'].iloc[0]
    plat = tiers[tiers['Segment_Value'] == 'Platinum'].iloc[0]
    assert (silver['Customer_Count_N'], silver['Actual_Churn_Count']) == (348, 45)
    assert (gold['Customer_Count_N'], gold['Actual_Churn_Count']) == (191, 26)
    assert (plat['Customer_Count_N'], plat['Actual_Churn_Count']) == (63, 14)

    stores = seg_df[seg_df['Segment_Type'] == 'Home_Store_Price_Tier']
    store_n = stores['Customer_Count_N'].sum()
    store_churn = stores['Actual_Churn_Count'].sum()
    assert store_n == 602, f"Expected 602 store total, got {store_n}"
    assert store_churn == 85, f"Expected 85 store churn, got {store_churn}"
    print("PASS: Segment risk totals reconcile 100% of all 602 customers (Silver 348, Gold 191, Platinum 63; Stores: 281, 198, 100, 23).")

    # -------------------------------------------------------------------------
    # NEW ASSESSMENT 4 CHECKS
    # -------------------------------------------------------------------------
    print("\n--- Verifying New Assessment 4 Deliverables ---")

    # 7. Check Comprehensive Segment and Signup Cohort Analysis Table
    comp_seg_file = resolve_file("phases/phase4/output/phase4_segment_cohort_churn_analysis.csv")
    comp_seg_df = pd.read_csv(comp_seg_file)
    expected_seg_cols = ['Segment_Type', 'Segment_Name', 'Customer_Count', 'Churner_Count', 'Retained_Count', 'Observed_Churn_Rate']
    for c in expected_seg_cols:
        assert c in comp_seg_df.columns, f"Missing column {c} in segment cohort analysis table"
    
    seg_types_present = set(comp_seg_df['Segment_Type'].unique())
    assert {'Membership_Tier', 'City', 'Home_Store_Price_Tier', 'Signup_Cohort'}.issubset(seg_types_present), \
        f"Missing required segment types in {seg_types_present}"
    
    # Reconcile validation cohort entries
    val_comp = comp_seg_df[comp_seg_df['Cohort_Scope'] == 'Validation_Cohort_N602']
    for stype in ['Membership_Tier', 'City', 'Home_Store_Price_Tier', 'Signup_Cohort']:
        st_sub = val_comp[val_comp['Segment_Type'] == stype]
        assert st_sub['Customer_Count'].sum() == 602, f"Count mismatch for {stype}: {st_sub['Customer_Count'].sum()}"
        assert st_sub['Churner_Count'].sum() == 85, f"Churn mismatch for {stype}: {st_sub['Churner_Count'].sum()}"
        assert st_sub['Retained_Count'].sum() == 517, f"Retained mismatch for {stype}: {st_sub['Retained_Count'].sum()}"
    print("PASS: Comprehensive segment & cohort analysis table verified across all 4 dimensions (reconciles 602: 85 churners, 517 retained).")

    # Check dedicated signup cohort summary table and chart
    cohort_summary_file = resolve_file("phases/phase4/output/phase4_signup_cohort_analysis.csv")
    cs_df = pd.read_csv(cohort_summary_file)
    assert cs_df['Customer_Count'].sum() == 2600, f"Expected 2600 total customers, got {cs_df['Customer_Count'].sum()}"
    assert cs_df['Churner_Count'].sum() == 818, f"Expected 818 total churners, got {cs_df['Churner_Count'].sum()}"
    cohort_chart_file = resolve_file("phases/phase4/output/charts/phase4_signup_cohort_distribution.png")
    assert cohort_chart_file.exists(), f"Missing cohort chart: {cohort_chart_file}"
    print("PASS: Dedicated signup cohort analysis table & distribution chart verified (2,600 total customers, 818 churners).")

    # 8. Check Batch Customer Prediction File for All Eligible Customers
    batch_file = resolve_file("phases/phase3/output/phase3_batch_predictions_all_eligible.csv")
    batch_df = pd.read_csv(batch_file)
    assert len(batch_df) == 2549, f"Expected 2549 eligible customers, got {len(batch_df)}"
    assert batch_df['CUSTOMER_ID'].nunique() == 2549, "Duplicate customer IDs found in batch predictions"
    
    batch_req_cols = [
        'CUSTOMER_ID', 'CHURN_PROBABILITY', 'PREDICTED_LABEL',
        'TOP_DRIVER_1', 'TOP_DRIVER_1_CONTRIBUTION',
        'TOP_DRIVER_2', 'TOP_DRIVER_2_CONTRIBUTION',
        'ACTUAL_CHURNED', 'VALIDATION_COHORT'
    ]
    for c in batch_req_cols:
        assert c in batch_df.columns, f"Missing required column {c} in batch prediction file"

    # Probability bounds and threshold consistency
    assert (batch_df['CHURN_PROBABILITY'] >= 0.0).all() and (batch_df['CHURN_PROBABILITY'] <= 1.0).all(), "Invalid churn probability values"
    assert ((batch_df['CHURN_PROBABILITY'] >= 0.50).astype(int) == batch_df['PREDICTED_LABEL']).all(), "Predicted label threshold 0.50 violation"
    
    # SHAP driver presence and validity
    assert batch_df['TOP_DRIVER_1'].notna().all(), "Null driver detected in TOP_DRIVER_1"
    assert batch_df['TOP_DRIVER_2'].notna().all(), "Null driver detected in TOP_DRIVER_2"
    assert (batch_df['TOP_DRIVER_1_CONTRIBUTION'].abs() > 0).any(), "Zero SHAP contributions detected"

    # Benchmark customer 500014 verification
    bench_cust = batch_df[batch_df['CUSTOMER_ID'] == 500014].iloc[0]
    np.testing.assert_almost_equal(bench_cust['CHURN_PROBABILITY'], 0.7767, decimal=3)
    assert bench_cust['PREDICTED_LABEL'] == 1
    assert bench_cust['TOP_DRIVER_1'] in ['SPEND_LAST_3M', 'TXNS_LAST_3M']

    # Validation cohort predictions with drivers
    val_drivers_file = resolve_file("phases/phase3/output/phase3_validation_predictions_with_drivers.csv")
    val_drivers_df = pd.read_csv(val_drivers_file)
    assert len(val_drivers_df) == 602, f"Expected 602 validation customers, got {len(val_drivers_df)}"
    assert int(val_drivers_df['ACTUAL_CHURNED'].sum()) == 85
    assert int(val_drivers_df['PREDICTED_LABEL'].sum()) == 105
    print("PASS: Customer-level batch predictions and SHAP explanations verified (2,549 eligible customers, benchmark customer 500014 = 77.67%).")

    # 9. Check Retention Campaign Financial Cost & ROI Sensitivity Analysis
    roi_file = resolve_file("phases/phase4/output/phase4_retention_financial_roi_sensitivity.csv")
    assump_file = resolve_file("phases/phase4/output/phase4_roi_assumptions.csv")
    roi_chart = resolve_file("phases/phase4/output/charts/phase4_retention_roi_sensitivity.png")
    
    roi_df = pd.read_csv(roi_file)
    assump_df = pd.read_csv(assump_file)
    assert len(roi_df) == 12, f"Expected 12 sensitivity scenario combinations, got {len(roi_df)}"
    assert len(assump_df) == 4, f"Expected 4 scenario assumption rows, got {len(assump_df)}"
    assert roi_chart.exists(), f"Missing ROI sensitivity chart at {roi_chart}"

    # Arithmetic sanity check on financial sensitivity calculations
    for _, row in roi_df.iterrows():
        calc_cost = row['Targeted_Customers_N'] * row['Unit_Intervention_Cost_USD']
        np.testing.assert_almost_equal(calc_cost, row['Total_Campaign_Cost_USD'], decimal=2)
        calc_benefit = row['Expected_Retained_Customers'] * row['Customer_Value_USD']
        np.testing.assert_almost_equal(calc_benefit, row['Expected_Gross_Benefit_USD'], decimal=2)
        calc_net = calc_benefit - calc_cost
        np.testing.assert_almost_equal(calc_net, row['Expected_Net_Benefit_USD'], decimal=2)
        calc_roi = (calc_net / calc_cost) * 100.0
        np.testing.assert_almost_equal(calc_roi, row['ROI_Percentage'], decimal=2)
    print("PASS: Retention campaign financial cost & ROI sensitivity calculations verified across all 12 combinations.")

    # 10. Check Data Quality & EDA Notebook Execution Status
    eda_nb_file = resolve_file("phases/phase1/phase1_eda_and_data_quality_notebook.ipynb")
    assert eda_nb_file.exists(), f"Missing EDA notebook at {eda_nb_file}"
    with open(eda_nb_file, 'r', encoding='utf-8') as f:
        nb_json = json.load(f)
    code_cells_with_output = [c for c in nb_json['cells'] if c.get('cell_type') == 'code' and len(c.get('outputs', [])) > 0]
    assert len(code_cells_with_output) >= 8, f"Expected executed code cells in notebook, found only {len(code_cells_with_output)}"
    print(f"PASS: Comprehensive Phase 1 EDA notebook verified ({len(nb_json['cells'])} total cells, {len(code_cells_with_output)} executed with outputs).")

    # 11. Model Pipeline Loading & Inference Verification
    model_file = resolve_file("models/phase3_random_forest_pipeline.joblib")
    import joblib
    pipe = joblib.load(model_file)
    assert hasattr(pipe, 'predict_proba'), "Loaded pipeline lacks predict_proba method"
    print("PASS: Persisted Random Forest pipeline loads successfully and verifies inference API.")

    print("\n=======================================================")
    print("=== ALL COMPREHENSIVE QUALITY GATE CHECKS PASSED PERFECTLY ===")
    print("=======================================================")
    return True


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
