"""
Automated unit and integration tests for Assessment 4 new deliverables:
1. Batch prediction and customer-level SHAP drivers
2. Segment and signup cohort churn analysis & reconciliation
3. Retention campaign financial cost & ROI sensitivity calculations
4. EDA notebook deliverables
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class TestAssessment4Deliverables(unittest.TestCase):

    def test_batch_predictions_all_eligible(self):
        """Verify batch prediction file for all 2,549 eligible customers."""
        batch_path = PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_batch_predictions_all_eligible.csv"
        self.assertTrue(batch_path.exists(), f"Missing {batch_path}")
        
        df = pd.read_csv(batch_path)
        # Check row count
        self.assertEqual(len(df), 2549, f"Expected 2549 eligible customers, got {len(df)}")
        # Check unique customer IDs
        self.assertEqual(df['CUSTOMER_ID'].nunique(), 2549, "Duplicate customer IDs found")
        
        # Check required columns
        expected_cols = [
            'CUSTOMER_ID', 'CHURN_PROBABILITY', 'PREDICTED_LABEL',
            'TOP_DRIVER_1', 'TOP_DRIVER_1_CONTRIBUTION',
            'TOP_DRIVER_2', 'TOP_DRIVER_2_CONTRIBUTION',
            'ACTUAL_CHURNED', 'VALIDATION_COHORT'
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing required column {col}")

        # Check probability bounds
        self.assertTrue((df['CHURN_PROBABILITY'] >= 0.0).all())
        self.assertTrue((df['CHURN_PROBABILITY'] <= 1.0).all())

        # Check threshold consistency (tau = 0.50)
        expected_labels = (df['CHURN_PROBABILITY'] >= 0.50).astype(int)
        np.testing.assert_array_equal(df['PREDICTED_LABEL'].values, expected_labels.values)

        # Check benchmark customer 500014
        cust_row = df[df['CUSTOMER_ID'] == 500014]
        self.assertEqual(len(cust_row), 1)
        prob_500014 = cust_row.iloc[0]['CHURN_PROBABILITY']
        self.assertAlmostEqual(prob_500014, 0.7767, delta=0.01)
        self.assertEqual(cust_row.iloc[0]['PREDICTED_LABEL'], 1)

        # Check drivers are populated with non-zero contributions
        self.assertTrue(df['TOP_DRIVER_1'].notna().all())
        self.assertTrue(df['TOP_DRIVER_2'].notna().all())
        self.assertTrue((df['TOP_DRIVER_1_CONTRIBUTION'].abs() > 0).any())
        
        # Verify drivers are distinct for each customer
        self.assertTrue((df['TOP_DRIVER_1'] != df['TOP_DRIVER_2']).all(), "Driver 1 and Driver 2 must be distinct")

        # Verify driver names are valid model features, not IDs, target labels, or admin fields
        from predict_customer import load_pipeline
        from phases.phase3.batch_predict import NUM_COLUMNS, CAT_COLUMNS
        pipe = load_pipeline()
        ohe = pipe.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
        valid_features = set(NUM_COLUMNS + ohe.get_feature_names_out(CAT_COLUMNS).tolist())
        self.assertTrue(set(df['TOP_DRIVER_1']).issubset(valid_features), "Invalid feature in TOP_DRIVER_1")
        self.assertTrue(set(df['TOP_DRIVER_2']).issubset(valid_features), "Invalid feature in TOP_DRIVER_2")

    def test_validation_predictions_with_drivers(self):
        """Verify validation cohort predictions with drivers has 602 rows and reconciles."""
        val_path = PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions_with_drivers.csv"
        self.assertTrue(val_path.exists(), f"Missing {val_path}")
        df = pd.read_csv(val_path)
        self.assertEqual(len(df), 602)
        self.assertEqual(int(df['ACTUAL_CHURNED'].sum()), 85)
        self.assertEqual(int(df['PREDICTED_LABEL'].sum()), 105)

    def test_segment_cohort_analysis(self):
        """Verify comprehensive segment and signup cohort churn analysis table."""
        seg_path = PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_segment_cohort_churn_analysis.csv"
        self.assertTrue(seg_path.exists(), f"Missing {seg_path}")
        df = pd.read_csv(seg_path)

        # Check columns
        req_cols = ['Segment_Type', 'Segment_Name', 'Customer_Count', 'Churner_Count', 'Retained_Count', 'Observed_Churn_Rate']
        for col in req_cols:
            self.assertIn(col, df.columns)

        # Check arithmetic reconciliation for every single row
        self.assertTrue((df['Customer_Count'] == df['Churner_Count'] + df['Retained_Count']).all())
        expected_rates = (df['Churner_Count'] / df['Customer_Count'] * 100).round(2)
        np.testing.assert_array_almost_equal(df['Observed_Churn_Rate'].values, expected_rates.values, decimal=2)

        # Check 4 segment types exist
        seg_types = set(df['Segment_Type'].unique())
        self.assertIn('Membership_Tier', seg_types)
        self.assertIn('City', seg_types)
        self.assertIn('Home_Store_Price_Tier', seg_types)
        self.assertIn('Signup_Cohort', seg_types)

        # Reconcile validation cohort entries
        val_entries = df[df['Cohort_Scope'] == 'Validation_Cohort_N602']
        for stype in ['Membership_Tier', 'City', 'Home_Store_Price_Tier', 'Signup_Cohort']:
            sub = val_entries[val_entries['Segment_Type'] == stype]
            self.assertEqual(sub['Customer_Count'].sum(), 602, f"Count mismatch for {stype}")
            self.assertEqual(sub['Churner_Count'].sum(), 85, f"Churn mismatch for {stype}")
            self.assertEqual(sub['Retained_Count'].sum(), 517, f"Retained mismatch for {stype}")

        # Reconcile all eligible entries
        el_entries = df[df['Cohort_Scope'] == 'All_Eligible_N2549']
        for stype in ['Membership_Tier', 'City', 'Home_Store_Price_Tier', 'Signup_Cohort']:
            sub = el_entries[el_entries['Segment_Type'] == stype]
            self.assertEqual(sub['Customer_Count'].sum(), 2549, f"Eligible count mismatch for {stype}")
            self.assertEqual(sub['Churner_Count'].sum(), 818, f"Eligible churn mismatch for {stype}")
            self.assertEqual(sub['Retained_Count'].sum(), 1731, f"Eligible retained mismatch for {stype}")

        # Check dedicated cohort table and chart
        cohort_csv = PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_signup_cohort_analysis.csv"
        cohort_png = PROJECT_ROOT / "phases" / "phase4" / "output" / "charts" / "phase4_signup_cohort_distribution.png"
        self.assertTrue(cohort_csv.exists())
        self.assertTrue(cohort_png.exists())
        cdf = pd.read_csv(cohort_csv)
        self.assertEqual(cdf['Customer_Count'].sum(), 2600)
        self.assertEqual(cdf['Churner_Count'].sum(), 818)

    def test_retention_financial_roi_sensitivity(self):
        """Verify financial cost, benefit, and ROI sensitivity calculations."""
        roi_csv = PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_retention_financial_roi_sensitivity.csv"
        assump_csv = PROJECT_ROOT / "phases" / "phase4" / "output" / "phase4_roi_assumptions.csv"
        roi_png = PROJECT_ROOT / "phases" / "phase4" / "output" / "charts" / "phase4_retention_roi_sensitivity.png"
        
        self.assertTrue(roi_csv.exists())
        self.assertTrue(assump_csv.exists())
        self.assertTrue(roi_png.exists())

        roi_df = pd.read_csv(roi_csv)
        self.assertEqual(len(roi_df), 12)  # 4 scenarios * 3 efficacy rates

        for _, row in roi_df.iterrows():
            # Arithmetic verification
            exp_cost = row['Targeted_Customers_N'] * row['Unit_Intervention_Cost_USD']
            self.assertAlmostEqual(exp_cost, row['Total_Campaign_Cost_USD'], delta=0.01)

            exp_benefit = row['Expected_Retained_Customers'] * row['Customer_Value_USD']
            self.assertAlmostEqual(exp_benefit, row['Expected_Gross_Benefit_USD'], delta=0.01)

            exp_net = exp_benefit - exp_cost
            self.assertAlmostEqual(exp_net, row['Expected_Net_Benefit_USD'], delta=0.01)

            exp_roi = (exp_net / exp_cost) * 100.0
            self.assertAlmostEqual(exp_roi, row['ROI_Percentage'], delta=0.05)

    def test_eda_notebook_exists_and_executed(self):
        """Verify that genuine EDA and Cleaning notebooks exist, have executed outputs, and zero errors."""
        import json
        for nb_rel in [
            "phases/phase1/phase1_eda_and_data_quality_notebook.ipynb",
            "phases/phase1/phase1_cleaning_notebook.ipynb"
        ]:
            nb_path = PROJECT_ROOT / nb_rel
            self.assertTrue(nb_path.exists(), f"Notebook missing: {nb_path}")
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
            self.assertGreater(len(code_cells), 5, f"Too few code cells in {nb_rel}")
            for idx, c in enumerate(code_cells):
                self.assertGreater(len(c.get('outputs', [])), 0, f"Unexecuted code cell {idx} in {nb_rel}")
                for out in c.get('outputs', []):
                    self.assertNotEqual(out.get('output_type'), 'error', f"Execution error in cell {idx} of {nb_rel}: {out.get('evalue')}")


if __name__ == '__main__':
    unittest.main()
