"""
Centralized path resolution utility for FreshBasket Churn Prediction project.
Dynamically resolves paths relative to the project root directory.
"""

from pathlib import Path

# Resolve project root dynamically (two levels up from phases/paths.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Core Project Directories
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
PHASES_DIR = PROJECT_ROOT / "phases"

# Phase Directories
PHASE1_DIR = PHASES_DIR / "phase1"
PHASE1_OUTPUT = PHASE1_DIR / "output"
PHASE1_CHARTS = PHASE1_OUTPUT / "charts"
PHASE1_CLEANED = PHASE1_OUTPUT / "cleaned_data"

PHASE2_DIR = PHASES_DIR / "phase2"
PHASE2_OUTPUT = PHASE2_DIR / "output"

PHASE3_DIR = PHASES_DIR / "phase3"
PHASE3_OUTPUT = PHASE3_DIR / "output"

PHASE4_DIR = PHASES_DIR / "phase4"
PHASE4_OUTPUT = PHASE4_DIR / "output"
PHASE4_CHARTS = PHASE4_OUTPUT / "charts"

# Key Artifacts
RAW_EXCEL_PATH = DATA_RAW_DIR / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
PROCESSED_DATASET_PATH = DATA_PROCESSED_DIR / "phase2_modeling_dataset.csv"
SAVED_MODEL_PATH = MODELS_DIR / "phase3_random_forest_pipeline.joblib"

# Phase 1 Cleaned Tables
CLEANED_CUSTOMER_PROFILE_PATH = DATA_INTERIM_DIR / "cleaned_fb_customer_profile.csv"
CLEANED_CHURN_LABEL_PATH = DATA_INTERIM_DIR / "cleaned_fb_churn_label.csv"
CLEANED_MONTHLY_ACTIVITY_PATH = DATA_INTERIM_DIR / "cleaned_fb_monthly_activity.csv"

# Phase 3 Output Artifacts
VALIDATION_PREDS_PATH = PHASE3_OUTPUT / "phase3_validation_predictions.csv"
MODEL_METRICS_PATH = PHASE3_OUTPUT / "phase3_model_metrics.csv"
LR_COEFS_PATH = PHASE3_OUTPUT / "phase3_logistic_regression_coefficients.csv"
RF_IMPORTANCES_PATH = PHASE3_OUTPUT / "phase3_random_forest_feature_importances.csv"

# Phase 4 Output Artifacts
TOP_PREDICTORS_PATH = PHASE4_OUTPUT / "phase4_step1_top_predictors.csv"
SHAP_SUMMARY_PATH = PHASE4_OUTPUT / "phase4_step1_shap_summary.csv"
SEGMENT_RISK_SUMMARY_PATH = PHASE4_OUTPUT / "phase4_step1_segment_risk_summary.csv"
CITY_RISK_SUMMARY_PATH = PHASE4_OUTPUT / "phase4_step1_city_risk_summary.csv"
ABLATION_RESULTS_PATH = PHASE4_OUTPUT / "phase4_step2_ablation_results.csv"
ABLATION_DELTAS_PATH = PHASE4_OUTPUT / "phase4_step2_ablation_deltas.csv"
RETENTION_SCENARIO_RESULTS_PATH = PHASE4_OUTPUT / "phase4_step3_retention_scenario_results.csv"
SCENARIO_OVERLAP_PATH = PHASE4_OUTPUT / "phase4_step3_scenario_overlap.csv"
CUSTOMER_TARGETING_ROSTER_PATH = PHASE4_OUTPUT / "phase4_step3_customer_targeting_roster.csv"

# Config files
REQUIREMENTS_PATH = CONFIG_DIR / "requirements.txt"
GITIGNORE_PATH = CONFIG_DIR / ".gitignore"

def get_project_root() -> Path:
    """Return the absolute Path to the project root directory."""
    return PROJECT_ROOT
