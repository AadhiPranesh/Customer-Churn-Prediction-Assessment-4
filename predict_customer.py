"""
FreshBasket Customer Churn Prediction — Single Customer Predictor
Tailwyndz Technologies Propel Internship Assessment 4

This standalone entry point loads the trained Random Forest pipeline from 'models/phase3_random_forest_pipeline.joblib'
and predicts churn risk for a new loyalty member based on interactive terminal inputs.

Usage:
    python predict_customer.py           # Interactive customer entry
    python predict_customer.py --verify  # Run validation benchmark test on Customer 500014
"""

import sys
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Safe project root determination (always relative to this script file)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Primary canonical paths
SAVED_MODEL_PATH = PROJECT_ROOT / "models" / "phase3_random_forest_pipeline.joblib"
PROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv"
VALIDATION_PREDS_PATH = PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions.csv"

# Fallback path checking helper
def _find_file(preferred_path, alternative_paths):
    preferred = Path(preferred_path)
    if preferred.exists():
        return preferred
    for alt in alternative_paths:
        p = Path(alt)
        if p.exists():
            return p
    return preferred

PIPELINE_PATH = _find_file(
    SAVED_MODEL_PATH,
    [
        PROJECT_ROOT / "models" / "phase3_random_forest_pipeline.joblib",
        PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_random_forest_pipeline.joblib",
        PROJECT_ROOT / "phase3_random_forest_pipeline.joblib"
    ]
)
VERIFICATION_DATASET_PATH = _find_file(
    PROCESSED_DATASET_PATH,
    [
        PROJECT_ROOT / "data" / "processed" / "phase2_modeling_dataset.csv",
        PROJECT_ROOT / "phases" / "phase2" / "output" / "phase2_modeling_dataset.csv",
        PROJECT_ROOT / "phase2_modeling_dataset.csv"
    ]
)
VERIFICATION_PREDS_PATH = _find_file(
    VALIDATION_PREDS_PATH,
    [
        PROJECT_ROOT / "phases" / "phase3" / "output" / "phase3_validation_predictions.csv",
        PROJECT_ROOT / "outputs" / "evaluation" / "phase3_validation_predictions.csv",
        PROJECT_ROOT / "phase3_validation_predictions.csv"
    ]
)

# Exact 28 feature columns expected by the trained pipeline
FEATURE_COLUMNS = [
    'AGE', 'AGE_IS_MISSING', 'GENDER', 'CITY', 'MEMBERSHIP_TIER',
    'HOME_STORE_PRICE_TIER', 'ELIGIBLE_PRE_CUTOFF_MONTHS', 'TENURE_DAYS_PRE_CUTOFF',
    'TOTAL_TXNS_PRE_CUTOFF', 'TOTAL_SPEND_PRE_CUTOFF', 'ACTIVE_MONTHS_PRE_CUTOFF',
    'PRE_CUTOFF_OBSERVED_MONTHS', 'TOTAL_APP_SESSIONS_PRE_CUTOFF',
    'TOTAL_EMAILS_OPENED_PRE_CUTOFF', 'TOTAL_COUPONS_REDEEMED_PRE_CUTOFF',
    'TOTAL_SUPPORT_TICKETS_PRE_CUTOFF', 'COMPLAINT_RECORDED_PRE_CUTOFF',
    'TXNS_LAST_3M', 'SPEND_LAST_3M', 'RECENCY_MONTHS_PRE_CUTOFF',
    'HAS_NO_PRE_CUTOFF_ACTIVITY', 'AVG_MONTHLY_TXNS_PRE_CUTOFF',
    'AVG_MONTHLY_SPEND_PRE_CUTOFF', 'AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF',
    'AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF', 'AVG_BASKET_VALUE_PRE_CUTOFF',
    'NEVER_PURCHASED_PRE_CUTOFF', 'IS_LIMITED_PRE_CUTOFF_HISTORY'
]

# Accepted values for categoricals
VALID_GENDERS = {'F': 'F', 'M': 'M', 'FEMALE': 'F', 'MALE': 'M', 'OTHER': 'Other', 'UNKNOWN': 'Unknown'}
VALID_TIERS = {'SILVER': 'Silver', 'GOLD': 'Gold', 'PLATINUM': 'Platinum'}
VALID_STORE_TIERS = {'MAINSTREAM': 'Mainstream', 'VALUE': 'Value', 'UPSCALE': 'Upscale', 'UNKNOWN': 'Unknown'}
VALID_CITIES = ['Madison', 'Spokane', 'Tulsa', 'Portland', 'Nashville', 'Boise', 'Austin', 'Columbus', 'Denver', 'Raleigh']


def load_pipeline():
    """Load the persisted Random Forest model pipeline."""
    if not Path(PIPELINE_PATH).exists():
        print(f"Error: Model pipeline file '{PIPELINE_PATH}' not found.")
        print("Please ensure you are running this script in the root project folder or appropriate environment.")
        sys.exit(1)
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        return pipeline
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        sys.exit(1)


def prompt_input(prompt_text, default_val=None, val_type=str, min_val=None, max_val=None, allowed_set=None):
    """Robust prompt helper with validation, defaults, and type casting."""
    while True:
        default_str = f" [Press Enter for: {default_val}]" if default_val is not None else ""
        raw_val = input(f"{prompt_text}{default_str}: ").strip()
        
        if not raw_val:
            if default_val is not None:
                return default_val
            print("  -> Input cannot be empty. Please enter a valid value.")
            continue
            
        try:
            if val_type == int:
                val = int(raw_val)
            elif val_type == float:
                val = float(raw_val)
            else:
                val = str(raw_val)
                
            if min_val is not None and val < min_val:
                print(f"  -> Value must be at least {min_val}. You entered: {val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  -> Value cannot exceed {max_val}. You entered: {val}")
                continue
            if allowed_set is not None:
                upper_key = str(val).strip().upper()
                if upper_key in allowed_set:
                    return allowed_set[upper_key] if isinstance(allowed_set, dict) else val
                else:
                    options_str = ", ".join(allowed_set.keys() if isinstance(allowed_set, dict) else allowed_set)
                    print(f"  -> Invalid choice. Accepted options: {options_str}")
                    continue
                    
            return val
        except ValueError:
            print(f"  -> Invalid format. Expected {val_type.__name__}.")


def derive_features(raw_inputs):
    """
    Construct the full 28-feature dictionary using exact business logic.
    """
    # 1. Age and missingness flag
    age = raw_inputs['AGE']
    if age is None or np.isnan(age):
        age_val = np.nan
        age_is_missing = 1
    else:
        age_val = float(age)
        age_is_missing = 0
        
    eligible_months = int(raw_inputs['ELIGIBLE_PRE_CUTOFF_MONTHS'])
    tenure_days = int(raw_inputs['TENURE_DAYS_PRE_CUTOFF'])
    total_txns = int(raw_inputs['TOTAL_TXNS_PRE_CUTOFF'])
    total_spend = round(float(raw_inputs['TOTAL_SPEND_PRE_CUTOFF']), 2)
    active_months = int(raw_inputs['ACTIVE_MONTHS_PRE_CUTOFF'])
    observed_months = int(raw_inputs['PRE_CUTOFF_OBSERVED_MONTHS'])
    app_sessions = int(raw_inputs['TOTAL_APP_SESSIONS_PRE_CUTOFF'])
    emails_opened = int(raw_inputs['TOTAL_EMAILS_OPENED_PRE_CUTOFF'])
    coupons = int(raw_inputs['TOTAL_COUPONS_REDEEMED_PRE_CUTOFF'])
    tickets = int(raw_inputs['TOTAL_SUPPORT_TICKETS_PRE_CUTOFF'])
    complaint = int(raw_inputs['COMPLAINT_RECORDED_PRE_CUTOFF'])
    txns_last_3m = int(raw_inputs['TXNS_LAST_3M'])
    spend_last_3m = round(float(raw_inputs['SPEND_LAST_3M']), 2)
    recency_months = int(raw_inputs['RECENCY_MONTHS_PRE_CUTOFF'])
    
    # 2. Activity indicator
    has_no_activity = 1 if observed_months == 0 else 0
    
    # 3. Monthly rates normalized by ELIGIBLE_PRE_CUTOFF_MONTHS
    if eligible_months > 0:
        avg_monthly_txns = round(total_txns / eligible_months, 2)
        avg_monthly_spend = round(total_spend / eligible_months, 2)
        avg_monthly_app = round(app_sessions / eligible_months, 2)
        avg_monthly_emails = round(emails_opened / eligible_months, 2)
    else:
        avg_monthly_txns = 0.0
        avg_monthly_spend = 0.0
        avg_monthly_app = 0.0
        avg_monthly_emails = 0.0

    # 4. Average basket value
    if total_txns > 0:
        avg_basket_value = round(total_spend / total_txns, 2)
    else:
        avg_basket_value = 0.0

    # 5. Never purchased flag
    never_purchased = 1 if (recency_months >= 99 or total_txns == 0) else 0

    # 6. Limited history flag
    is_limited_history = 1 if (tenure_days < 90 or observed_months < 4) else 0

    feature_dict = {
        'AGE': age_val,
        'AGE_IS_MISSING': age_is_missing,
        'GENDER': raw_inputs['GENDER'],
        'CITY': raw_inputs['CITY'],
        'MEMBERSHIP_TIER': raw_inputs['MEMBERSHIP_TIER'],
        'HOME_STORE_PRICE_TIER': raw_inputs['HOME_STORE_PRICE_TIER'],
        'ELIGIBLE_PRE_CUTOFF_MONTHS': eligible_months,
        'TENURE_DAYS_PRE_CUTOFF': tenure_days,
        'TOTAL_TXNS_PRE_CUTOFF': total_txns,
        'TOTAL_SPEND_PRE_CUTOFF': total_spend,
        'ACTIVE_MONTHS_PRE_CUTOFF': active_months,
        'PRE_CUTOFF_OBSERVED_MONTHS': observed_months,
        'TOTAL_APP_SESSIONS_PRE_CUTOFF': app_sessions,
        'TOTAL_EMAILS_OPENED_PRE_CUTOFF': emails_opened,
        'TOTAL_COUPONS_REDEEMED_PRE_CUTOFF': coupons,
        'TOTAL_SUPPORT_TICKETS_PRE_CUTOFF': tickets,
        'COMPLAINT_RECORDED_PRE_CUTOFF': complaint,
        'TXNS_LAST_3M': txns_last_3m,
        'SPEND_LAST_3M': spend_last_3m,
        'RECENCY_MONTHS_PRE_CUTOFF': recency_months,
        'HAS_NO_PRE_CUTOFF_ACTIVITY': has_no_activity,
        'AVG_MONTHLY_TXNS_PRE_CUTOFF': avg_monthly_txns,
        'AVG_MONTHLY_SPEND_PRE_CUTOFF': avg_monthly_spend,
        'AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF': avg_monthly_app,
        'AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF': avg_monthly_emails,
        'AVG_BASKET_VALUE_PRE_CUTOFF': avg_basket_value,
        'NEVER_PURCHASED_PRE_CUTOFF': never_purchased,
        'IS_LIMITED_PRE_CUTOFF_HISTORY': is_limited_history
    }
    
    # Return as DataFrame matching the exact order
    df_features = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]
    return df_features, feature_dict


def run_verification_check(pipeline):
    """
    Self-verification check against customer 500014 from phase3_validation_predictions.csv.
    Expected Random Forest churn probability: ~77.67%.
    """
    print("\n" + "="*80)
    print("      AUTOMATED BENCHMARK VERIFICATION (Customer ID: 500014)")
    print("="*80)
    
    if not Path(VERIFICATION_DATASET_PATH).exists() or not Path(VERIFICATION_PREDS_PATH).exists():
        print(f"Verification files not found ({VERIFICATION_DATASET_PATH}, {VERIFICATION_PREDS_PATH}). Skipping benchmark verification.")
        return False
        
    df_dataset = pd.read_csv(VERIFICATION_DATASET_PATH)
    df_preds = pd.read_csv(VERIFICATION_PREDS_PATH)
    
    cid = 500014
    row = df_dataset[df_dataset['CUSTOMER_ID'] == cid]
    if len(row) == 0:
        print(f"Customer {cid} not found in {VERIFICATION_DATASET_PATH}.")
        return False
        
    saved_pred_row = df_preds[df_preds['CUSTOMER_ID'] == cid].iloc[0]
    expected_prob = float(saved_pred_row['RF_CHURN_PROBABILITY']) * 100
    expected_label = int(saved_pred_row['RF_PRED_LABEL'])
    
    X_cust = row[FEATURE_COLUMNS]
    live_prob = pipeline.predict_proba(X_cust)[0][1] * 100
    churn_probability = live_prob / 100.0
    predicted_class = int(churn_probability >= 0.50)
    
    print(f"Target Benchmark Customer:     {cid}")
    print(f"Actual Observed Churn:         {int(saved_pred_row['CHURNED'])} (1 = Churned in Q2 2024)")
    print(f"Saved Validation Probability:  {expected_prob:.2f}%")
    print(f"Fresh Computed Probability:    {live_prob:.2f}%")
    print(f"Predicted Churn Class:         {predicted_class}")
    print(f"Model Prediction Label:        {predicted_class} (Threshold = 0.50)")
    
    diff = abs(live_prob - expected_prob)
    if diff < 0.01:
        print("\n>>> BENCHMARK VERIFICATION SUCCESSFUL: Live pipeline matches saved results (77.67%)! <<<")
    else:
        print(f"\n>>> WARNING: Discrepancy of {diff:.4f}% detected compared to saved validation predictions. <<<")
    print("="*80 + "\n")
    return diff < 0.01


def collect_customer_inputs():
    """Interactively collect raw customer parameters."""
    print("\n" + "="*80)
    print("   FRESHBASKET RETAIL LOYALTY PROGRAM — CUSTOMER CHURN PREDICTOR")
    print("="*80)
    print("Please enter the customer details below. Reasonable defaults are provided in brackets.\n")

    # 1. Demographics
    print("--- 1. MEMBER DEMOGRAPHICS & STORE ---")
    age_str = input("Member Age (years, or press Enter if missing): ").strip()
    if not age_str:
        age = None
    else:
        try:
            age = float(age_str)
        except ValueError:
            print("  -> Invalid age entered. Defaulting to missing.")
            age = None

    gender = prompt_input("Gender (M / F / Other / Unknown)", default_val="F", allowed_set=VALID_GENDERS)
    city = prompt_input("City Name", default_val="Austin")
    tier = prompt_input("Membership Tier (Silver / Gold / Platinum)", default_val="Silver", allowed_set=VALID_TIERS)
    store_tier = prompt_input("Home Store Price Tier (Mainstream / Value / Upscale / Unknown)", default_val="Mainstream", allowed_set=VALID_STORE_TIERS)

    # 2. Tenure & Exposure
    print("\n--- 2. MEMBERSHIP TENURE & OBSERVATION WINDOW ---")
    tenure_days = prompt_input("Tenure (days enrolled prior to March 31, 2024)", default_val=180, val_type=int, min_val=0)
    eligible_months = prompt_input("Eligible Pre-Cutoff Months (1 to 15 calendar months)", default_val=6, val_type=int, min_val=1, max_val=15)
    observed_months = prompt_input("Observed Activity Months (count of monthly activity records, 0 to 15)", default_val=min(6, eligible_months), val_type=int, min_val=0, max_val=eligible_months)

    # 3. Transaction & Spend History
    print("\n--- 3. HISTORICAL TRANSACTIONS & SPENDING ---")
    total_txns = prompt_input("Total Transactions (pre-cutoff)", default_val=8, val_type=int, min_val=0)
    total_spend = prompt_input("Total Spend in Dollars ($) (pre-cutoff)", default_val=240.0, val_type=float, min_val=0.0)
    active_months = prompt_input("Active Shopping Months (months with at least 1 purchase)", default_val=min(total_txns, observed_months), val_type=int, min_val=0, max_val=observed_months)

    # 4. Recent Momentum & Recency
    print("\n--- 4. RECENT VISITS, SPEND & RECENCY ---")
    txns_3m = prompt_input("Transactions in Last 3 Months (Jan-Mar 2024)", default_val=min(total_txns, 3), val_type=int, min_val=0, max_val=total_txns)
    spend_3m = prompt_input("Spend in Last 3 Months ($) (Jan-Mar 2024)", default_val=min(total_spend, 95.0), val_type=float, min_val=0.0, max_val=total_spend)
    recency = prompt_input("Purchase Recency (calendar months since last purchase; 0=shopped in March, 1=Feb, 2=Jan, 99=never)", default_val=0 if txns_3m > 0 else 2, val_type=int, min_val=0)

    # 5. Support & Engagement
    print("\n--- 5. DIGITAL ENGAGEMENT & CUSTOMER SERVICE ---")
    app_sessions = prompt_input("Total Mobile App Sessions (pre-cutoff)", default_val=10, val_type=int, min_val=0)
    emails_opened = prompt_input("Total Promotional Emails Opened (pre-cutoff)", default_val=4, val_type=int, min_val=0)
    coupons = prompt_input("Total Loyalty Coupons Redeemed (pre-cutoff)", default_val=2, val_type=int, min_val=0)
    tickets = prompt_input("Total Customer Support Tickets (pre-cutoff)", default_val=0, val_type=int, min_val=0)
    complaint = prompt_input("Formal Complaint Logged? (1 = Yes, 0 = No)", default_val=0, val_type=int, min_val=0, max_val=1)

    raw_data = {
        'AGE': age,
        'GENDER': gender,
        'CITY': city,
        'MEMBERSHIP_TIER': tier,
        'HOME_STORE_PRICE_TIER': store_tier,
        'TENURE_DAYS_PRE_CUTOFF': tenure_days,
        'ELIGIBLE_PRE_CUTOFF_MONTHS': eligible_months,
        'PRE_CUTOFF_OBSERVED_MONTHS': observed_months,
        'TOTAL_TXNS_PRE_CUTOFF': total_txns,
        'TOTAL_SPEND_PRE_CUTOFF': total_spend,
        'ACTIVE_MONTHS_PRE_CUTOFF': active_months,
        'TXNS_LAST_3M': txns_3m,
        'SPEND_LAST_3M': spend_3m,
        'RECENCY_MONTHS_PRE_CUTOFF': recency,
        'TOTAL_APP_SESSIONS_PRE_CUTOFF': app_sessions,
        'TOTAL_EMAILS_OPENED_PRE_CUTOFF': emails_opened,
        'TOTAL_COUPONS_REDEEMED_PRE_CUTOFF': coupons,
        'TOTAL_SUPPORT_TICKETS_PRE_CUTOFF': tickets,
        'COMPLAINT_RECORDED_PRE_CUTOFF': complaint
    }
    return raw_data


def display_prediction_results(prob, pred_label, derived_dict):
    """Format and display prediction output with diagnostic explanations."""
    print("\n" + "="*80)
    print("                     PREDICTION & DIAGNOSTIC RESULTS")
    print("="*80)
    
    # Calculate churn probability and predicted class (Threshold = 0.50)
    churn_probability = prob / 100.0 if prob > 1.0 else prob
    predicted_class = int(churn_probability >= 0.50)
    classification_str = "PREDICTED CHURN (AT-RISK)" if predicted_class == 1 else "PREDICTED NOT CHURN (RETAINED)"
    
    # 1. Probability, Class, Classification & Threshold
    print(f"\n>>> PREDICTED CHURN PROBABILITY: {prob:.2f}% <<<")
    print(f">>> PREDICTED CHURN CLASS: {predicted_class} <<<")
    print(f">>> CLASSIFICATION: {classification_str} <<<")
    print(">>> DECISION THRESHOLD: 0.50 (50%) <<<")
    
    print("\n" + "-"*80)
    print("CLASS INTERPRETATION:")
    print("  • Class 0 = Predicted not churn (retained).")
    print("  • Class 1 = Predicted churn (at risk).")
    print("-"*80)
    
    # Risk Level
    if prob >= 70.0:
        risk_level = "CRITICAL RISK (Imminent Defection Likely)"
        playbook = "Deploy high-priority intervention (Scenario 1 Call / Scenario 2 Win-back Voucher)."
    elif prob >= 50.0:
        playbook = "Deploy proactive retention outreach (Scenario 2 Reactivation or Scenario 3 App Push)."
        risk_level = "ELEVATED RISK (Above 0.50 Action Threshold)"
    elif prob >= 30.0:
        risk_level = "MODERATE RISK (Monitor for Further Disengagement)"
        playbook = "Standard weekly loyalty engagement circular."
    else:
        risk_level = "LOW RISK (Active, Healthy Shopping Routine)"
        playbook = "Standard loyalty communications (no discounting needed)."
        
    print(f"\nRisk Assessment:   {risk_level}")
    print(f"Recommended Action: {playbook}")

    # 2. Key Diagnostic Signals
    print("\nKey Diagnostic Drivers for this Member:")
    rec = derived_dict['RECENCY_MONTHS_PRE_CUTOFF']
    q1_txns = derived_dict['TXNS_LAST_3M']
    tickets = derived_dict['TOTAL_SUPPORT_TICKETS_PRE_CUTOFF']
    complaint = derived_dict['COMPLAINT_RECORDED_PRE_CUTOFF']
    basket = derived_dict['AVG_BASKET_VALUE_PRE_CUTOFF']

    if rec >= 2:
        print(f"  [!] Purchase Dormancy: Member has not shopped for >= {rec} calendar months (strongest churn predictor).")
    else:
        print(f"  [*] High Recency Health: Member shopped recently (recency = {rec} months).")

    if q1_txns == 0:
        print("  [!] Zero Q1 Visit Velocity: No transactions recorded during the final 3 months.")
    else:
        print(f"  [*] Positive Visit Momentum: {q1_txns} transactions in the last 3 months.")

    if tickets > 0 or complaint == 1:
        print(f"  [!] Service Friction: {tickets} support ticket(s) and complaint flag = {complaint}.")
    else:
        print("  [*] Clean Service History: Zero recorded customer service friction.")

    print(f"  [*] Average Basket Value: ${basket:.2f} per shopping trip.")

    # 3. Essential Assessment Disclaimer
    print("\n" + "-"*80)
    print("SCIENTIFIC / METHODOLOGICAL NOTE:")
    print("  • This output is an empirical statistical prediction, NOT a guarantee of future behavior.")
    print("  • The model evaluates historical patterns prior to April 1, 2024 to estimate risk odds.")
    print("  • Actual customer retention requires actionable intervention and controlled A/B testing.")
    print("-"*80 + "\n")


def main():
    pipeline = load_pipeline()
    
    # Check if run with verification flag
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['--verify', '-v', 'verify']:
        success = run_verification_check(pipeline)
        sys.exit(0 if success else 1)

    while True:
        print("="*80)
        print("   FRESHBASKET CUSTOMER CHURN PREDICTOR (Random Forest Model)")
        print("="*80)
        print("Options:")
        print("  [1] Predict Churn for a New Customer (Interactive Entry)")
        print("  [2] Run Benchmark Verification Test on Customer 500014 (Expected: 77.67%)")
        print("  [3] Exit")
        
        choice = input("\nSelect an option [1/2/3, default = 1]: ").strip()
        
        if choice == '2':
            run_verification_check(pipeline)
            cont = input("\nWould you like to continue to interactive customer prediction? [y/N]: ").strip().lower()
            if cont not in ['y', 'yes']:
                print("Exiting.")
                return
            break
        elif choice == '3':
            print("Exiting.")
            return
        elif choice in ['1', '']:
            break
        else:
            print("Invalid option. Please enter 1, 2, or 3.")

    raw_inputs = collect_customer_inputs()
    X_new, derived_dict = derive_features(raw_inputs)
    
    # Predict
    prob = pipeline.predict_proba(X_new)[0][1] * 100.0
    churn_probability = prob / 100.0
    predicted_class = int(churn_probability >= 0.50)
    
    display_prediction_results(prob, predicted_class, derived_dict)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting.")
        sys.exit(0)
