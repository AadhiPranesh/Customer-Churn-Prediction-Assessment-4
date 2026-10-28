"""
Automated unit and regression tests for FreshBasket customer churn prediction.
Verifies model loading from models/, feature schema integrity, and benchmark customer 500014.
"""

import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phases.paths import (
    SAVED_MODEL_PATH,
    PROCESSED_DATASET_PATH,
    VALIDATION_PREDS_PATH
)
from predict_customer import (
    FEATURE_COLUMNS,
    derive_features,
    load_pipeline
)

class TestCustomerPrediction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_pipeline()
        cls.dataset = pd.read_csv(PROCESSED_DATASET_PATH)
        cls.val_preds = pd.read_csv(VALIDATION_PREDS_PATH)

    def test_pipeline_exists_and_loads(self):
        """Confirm pipeline file exists in models/ and loads successfully."""
        self.assertTrue(Path(SAVED_MODEL_PATH).exists(), f"Model missing at {SAVED_MODEL_PATH}")
        self.assertIsNotNone(self.pipeline)
        self.assertTrue(hasattr(self.pipeline, 'predict_proba'))

    def test_feature_columns_schema(self):
        """Confirm the exact 28 feature names match expectations."""
        self.assertEqual(len(FEATURE_COLUMNS), 28)
        self.assertIn('RECENCY_MONTHS_PRE_CUTOFF', FEATURE_COLUMNS)
        self.assertIn('TOTAL_TXNS_PRE_CUTOFF', FEATURE_COLUMNS)
        self.assertIn('IS_LIMITED_PRE_CUTOFF_HISTORY', FEATURE_COLUMNS)

    def test_benchmark_customer_500014(self):
        """
        Verify benchmark customer 500014 produces approximately 77.67% churn probability.
        """
        cid = 500014
        cust_row = self.dataset[self.dataset['CUSTOMER_ID'] == cid]
        self.assertEqual(len(cust_row), 1, f"Customer {cid} not found in dataset")
        
        saved_row = self.val_preds[self.val_preds['CUSTOMER_ID'] == cid].iloc[0]
        expected_prob = float(saved_row['RF_CHURN_PROBABILITY']) * 100.0
        
        X_cust = cust_row[FEATURE_COLUMNS]
        live_prob = self.pipeline.predict_proba(X_cust)[0][1] * 100.0
        
        self.assertAlmostEqual(live_prob, expected_prob, delta=0.01)
        self.assertAlmostEqual(live_prob, 77.67, delta=0.05)
        
        # Threshold 0.50
        live_pred = 1 if live_prob >= 50.0 else 0
        self.assertEqual(live_pred, 1)

    def test_feature_derivation_logic(self):
        """Confirm feature derivation matches exact business logic."""
        sample_input = {
            'AGE': 35.0,
            'GENDER': 'F',
            'CITY': 'Austin',
            'MEMBERSHIP_TIER': 'Gold',
            'HOME_STORE_PRICE_TIER': 'Mainstream',
            'TENURE_DAYS_PRE_CUTOFF': 200,
            'ELIGIBLE_PRE_CUTOFF_MONTHS': 6,
            'PRE_CUTOFF_OBSERVED_MONTHS': 6,
            'TOTAL_TXNS_PRE_CUTOFF': 12,
            'TOTAL_SPEND_PRE_CUTOFF': 360.0,
            'ACTIVE_MONTHS_PRE_CUTOFF': 6,
            'TXNS_LAST_3M': 6,
            'SPEND_LAST_3M': 180.0,
            'RECENCY_MONTHS_PRE_CUTOFF': 0,
            'TOTAL_APP_SESSIONS_PRE_CUTOFF': 15,
            'TOTAL_EMAILS_OPENED_PRE_CUTOFF': 8,
            'TOTAL_COUPONS_REDEEMED_PRE_CUTOFF': 3,
            'TOTAL_SUPPORT_TICKETS_PRE_CUTOFF': 0,
            'COMPLAINT_RECORDED_PRE_CUTOFF': 0
        }
        df_feat, feat_dict = derive_features(sample_input)
        self.assertEqual(df_feat.shape, (1, 28))
        self.assertEqual(feat_dict['AVG_MONTHLY_TXNS_PRE_CUTOFF'], 2.0)
        self.assertEqual(feat_dict['AVG_BASKET_VALUE_PRE_CUTOFF'], 30.0)
        self.assertEqual(feat_dict['HAS_NO_PRE_CUTOFF_ACTIVITY'], 0)
        self.assertEqual(feat_dict['NEVER_PURCHASED_PRE_CUTOFF'], 0)
        self.assertEqual(feat_dict['IS_LIMITED_PRE_CUTOFF_HISTORY'], 0)
        
        # Pipeline prediction works on derived features
        prob = self.pipeline.predict_proba(df_feat)[0][1] * 100.0
        self.assertTrue(0.0 <= prob <= 100.0)

    def test_interactive_prediction_execution(self):
        """Confirm predict_customer.py runs interactively without runtime errors."""
        import subprocess
        inputs = '1\n42\nM\nAustin\nGold\nMainstream\n250\n6\n6\n14\n420\n6\n6\n180\n0\n12\n8\n2\n0\n0\n'
        p = subprocess.run(
            [sys.executable, 'predict_customer.py'],
            input=inputs,
            text=True,
            capture_output=True,
            cwd=Path(__file__).resolve().parents[1]
        )
        self.assertEqual(p.returncode, 0, f"Interactive prediction failed: {p.stderr}")
        self.assertIn("PREDICTION & DIAGNOSTIC RESULTS", p.stdout)
        self.assertIn("PREDICTED CHURN PROBABILITY", p.stdout)


if __name__ == '__main__':
    unittest.main()

