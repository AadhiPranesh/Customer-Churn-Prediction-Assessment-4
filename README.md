# FreshBasket Customer Churn Prediction & Proactive Retention Engine
## Tailwyndz Technologies Propel Internship Assessment 4 — Reorganized Project Repository

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed%20%26%20Audited-brightgreen.svg)]()

This repository contains the complete, audited, and production-ready data science deliverables for the **FreshBasket Retail Loyalty Program Churn Prediction Assessment**. The project predicts customer defection for 2,600 loyalty members, attributes key risk drivers, evaluates feature domains via ablation, and simulates targeted operational retention playbooks without lookahead bias.

The repository has been safely refactored into a clean, professional, maintainable project structure following industry best practices.

---

## Table of Contents

1. [Executive Summary & Verified Key Results](#1-executive-summary--verified-key-results)
2. [Methodology & Safeguards](#2-methodology--safeguards)
3. [Reorganized Project Structure](#3-reorganized-project-structure)
4. [Environment Setup & Dependencies](#4-environment-setup--dependencies)
5. [Execution & Reproducibility Guide](#5-execution--reproducibility-guide)
   - [Interactive Customer Prediction](#interactive-customer-prediction)
   - [Automated Benchmark Verification](#automated-benchmark-verification)
   - [Comprehensive Quality Gate Audit](#comprehensive-quality-gate-audit)
   - [Automated Unit & Regression Tests](#automated-unit--regression-tests)
   - [Phase Pipeline Workflows](#phase-pipeline-workflows)
6. [Phase-by-Phase Deliverables Summary](#6-phase-by-phase-deliverables-summary)
7. [Project Report](#7-project-report)
8. [Project Presentation & Google Slides Deliverable](#8-project-presentation--google-slides-deliverable)
9. [Troubleshooting Guide](#9-troubleshooting-guide)

---

## 1. Executive Summary & Verified Key Results

Across 2,600 FreshBasket loyalty members, customer churn is evaluated over a quarterly target window (April 1 to June 30, 2024) using features engineered exclusively prior to April 1, 2024.

### Model Evaluation Benchmark (Validation Cohort: $N = 602$, 85 Churners, 517 Retained)

| Evaluation Metric | Logistic Regression (Baseline) | Random Forest Classifier | Operational Advantage |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **0.9468** | 0.9169 | LR shows higher overall agreement |
| **Precision** | **0.8955** | 0.6667 | LR yields only 7 false alarms (vs. 35 for RF) |
| **Recall (Sensitivity)** | 0.7059 | **0.8235** | **RF catches 70 of 85 churners (vs. 60 for LR)** |
| **F1-Score** | **0.7895** | 0.7368 | LR achieves superior harmonic balance |
| **PR-AUC (Precision-Recall)**| **0.8808** | 0.8499 | LR dominates precision-recall curve |
| **ROC-AUC** | **0.9644** | 0.9446 | Both models exhibit strong discrimination |

#### Confusion Matrices (Default Decision Threshold $\tau = 0.50$)

- **Logistic Regression Baseline:**
  $$\begin{bmatrix} TN = 510 & FP = 7 \\ FN = 25 & TP = 60 \end{bmatrix}$$
  *Interpretation:* Flagged 67 customers; 60 true churners caught, only 7 false positives. Highly cost-effective for expensive outreach.

- **Random Forest Classifier:**
  $$\begin{bmatrix} TN = 482 & FP = 35 \\ FN = 15 & TP = 70 \end{bmatrix}$$
  *Interpretation:* Flagged 105 customers; 70 true churners caught (only 15 missed), at the expense of 35 false alarms. Optimal for low-cost digital nudges.

---

## 2. Methodology & Safeguards

### Temporal Cutoff & Leakage Prevention
- **Cutoff Date:** `March 31, 2024, 23:59:59`.
- **Pre-Cutoff Predictor Window:** All behavioral records timestamped on or before March 31, 2024.
- **Post-Cutoff Target Window:** Churn label ($Y \in \{0, 1\}$) measured between April 1, 2024, and June 30, 2024.
- **Leakage Safeguard:** Zero post-cutoff activity is included in predictor features. Preprocessing transformers (imputers, scalers) are fit strictly on training cohort data.

### Cohort-Based Validation Design
Customers are split by enrollment date:
1. **Training Cohort ($N = 1,947$):** Signups on or before June 30, 2023. Tenures: 9 to 27 months. Historical churn rate: 37.65% (733 churners).
2. **Validation Cohort ($N = 602$):** Signups between July 1, 2023, and March 31, 2024. Tenures: 1 to 9 months. Churn rate: 14.12% (85 churners).
3. **Excluded Cohort ($N = 51$):** Signups after March 31, 2024. Excluded from all modeling due to having zero pre-cutoff history.

---

## 3. Final Project Structure

The project has been organized into a clean, professional, phase-based modular directory structure.
**The root directory contains strictly ONLY two files:** `predict_customer.py` and `README.md`.

```text
AADHI/
│
├── predict_customer.py                        # Standalone root predictor & benchmark launcher
├── README.md                                  # Main repository documentation & guide
│
├── config/                                    # Project configurations & environments
│   ├── requirements.txt                       # Verified project dependencies
│   └── .gitignore                             # Git ignore rules
│
├── data/                                      # Datasets organized by lifecycle stage
│   ├── raw/
│   │   └── FreshBasket_Loyalty_Churn_Dataset.xlsx   # Immutable raw source Excel dataset
│   ├── interim/
│   │   ├── cleaned_fb_churn_label.csv         # Cleaned interim churn labels
│   │   ├── cleaned_fb_customer_profile.csv    # Cleaned interim customer profile
│   │   └── cleaned_fb_monthly_activity.csv    # Cleaned interim monthly activity
│   └── processed/
│       └── phase2_modeling_dataset.csv        # Master modeling dataset (2,600 x 33)
│
├── models/
│   └── phase3_random_forest_pipeline.joblib   # Persisted fitted Random Forest pipeline
│
├── notebooks/                                 # Centralized Jupyter Notebooks
│   ├── phase1_eda_and_data_quality_notebook.ipynb # Comprehensive executed EDA notebook
│   ├── phase1_cleaning_notebook.ipynb         # Data cleaning & remediation notebook
│   ├── phase2_feature_engineering_notebook.ipynb # Feature derivation notebook
│   └── phase3_model_development_notebook.ipynb # Model benchmarking notebook
│
├── phases/                                    # Phase-based self-contained modules
│   ├── __init__.py
│   ├── paths.py                               # Centralized project path resolution
│   │
│   ├── phase1/                                # Phase 1: EDA, Data Cleaning & Schema Audit
│   │   ├── output/
│   │   │   ├── charts/                        # 7 publication charts (01 to 07)
│   │   │   └── cleaned_data/                  # Cleaned customer, label & activity tables
│   │   ├── generate_phase1_eda.py             # Phase 1 EDA chart generation script
│   │   ├── create_eda_notebook.py             # Script to build & execute EDA notebook
│   │   ├── phase1_eda_and_data_quality_notebook.ipynb # Executed comprehensive EDA notebook
│   │   ├── phase1_cleaning_notebook.ipynb     # Interactive data cleaning notebook
│   │   ├── phase1_data_quality_report.md      # Data quality audit report
│   │   ├── phase1_decisions.md                # Data cleaning decisions log
│   │   ├── phase1_eda_report.md               # Exploratory data analysis report
│   │   └── phase1_review_corrections.md       # Review corrections documentation
│   │
│   ├── phase2/                                # Phase 2: Feature Engineering & Dataset
│   │   ├── output/
│   │   │   └── phase2_modeling_dataset.csv    # Exported modeling dataset
│   │   ├── build_phase2_dataset.py            # Feature engineering dataset builder
│   │   ├── create_phase2_notebook.py          # Notebook generator
│   │   ├── phase2_feature_engineering_notebook.ipynb # Interactive feature engineering notebook
│   │   ├── phase2_feature_dictionary.md       # 28-feature dictionary & business definitions
│   │   ├── phase2_data_checks.md              # Feature distributions and sanity checks
│   │   ├── phase2_final_audit.md              # Phase 2 audit sign-off
│   │   └── phase2_validation_cohort_audit.md  # Temporal cohort partition audit
│   │
│   ├── phase3/                                # Phase 3: Baseline & Random Forest Modeling
│   │   ├── output/
│   │   │   ├── phase3_validation_predictions.csv  # 602 customer validation predictions
│   │   │   ├── phase3_validation_predictions_with_drivers.csv # Validation predictions with SHAP drivers
│   │   │   ├── phase3_batch_predictions_all_eligible.csv # Batch predictions for all 2,549 eligible members
│   │   │   ├── phase3_model_metrics.csv       # LR and RF model comparison metrics
│   │   │   ├── phase3_threshold_comparison.csv # Threshold sensitivity analysis (0.50 vs training CV)
│   │   │   ├── phase3_bootstrap_confidence_intervals.csv # 1,000-resample 95% bootstrap CIs
│   │   │   ├── phase3_logistic_regression_coefficients.csv # LR log-odds table
│   │   │   └── phase3_random_forest_feature_importances.csv # RF Gini importances
│   │   ├── batch_predict.py                   # Customer-level batch prediction & SHAP driver engine
│   │   ├── train_phase3_models.py             # Model training & validation evaluation script
│   │   ├── threshold_comparison.py            # Training-CV threshold optimization & sensitivity analysis
│   │   ├── bootstrap_confidence_intervals.py  # 1,000-resample bootstrap & paired PR-AUC difference test
│   │   ├── execute_phase3.py                  # End-to-end Phase 3 execution script
│   │   ├── create_phase3_notebook.py          # Notebook generation utility
│   │   ├── verify_phase3.py                   # Segment performance verifier
│   │   ├── audit_phase3_verification.py       # Metrics recalculation audit
│   │   ├── phase3_model_development_notebook.ipynb # Interactive model development notebook
│   │   └── phase3_model_evaluation_report.md  # Comprehensive model evaluation report
│   │
│   └── phase4/                                # Phase 4: Drivers, Ablation & Simulation
│       ├── output/
│       │   ├── charts/                        # Publication charts (SHAP, ablation, retention, ROI)
│       │   ├── phase4_step1_top_predictors.csv
│       │   ├── phase4_step1_shap_summary.csv
│       │   ├── phase4_step1_segment_risk_summary.csv
│       │   ├── phase4_step1_city_risk_summary.csv
│       │   ├── phase4_segment_cohort_churn_analysis.csv # Reconciled segments (Tier, City, Store, Cohort)
│       │   ├── phase4_signup_cohort_analysis.csv # 2,600 customer signup cohort breakdown
│       │   ├── phase4_step2_ablation_results.csv
│       │   ├── phase4_step2_ablation_deltas.csv
│       │   ├── phase4_step3_retention_scenario_results.csv
│       │   ├── phase4_step3_scenario_overlap.csv
│       │   ├── phase4_step3_customer_targeting_roster.csv # 105 at-risk customer roster
│       │   ├── phase4_roi_assumptions.csv     # Retention ROI assumption parameters
│       │   └── phase4_retention_financial_roi_sensitivity.csv # 12-scenario financial ROI sensitivity
│       ├── phase4_segment_cohort_analysis.py  # Segment & signup cohort churn analysis
│       ├── phase4_retention_roi_analysis.py   # Financial cost & ROI sensitivity analysis
│       ├── phase4_step1_driver_analysis.py    # SHAP & demographic segment profiling
│       ├── phase4_step2_ablation_study.py     # 4-subset feature ablation study
│       ├── phase4_step3_retention_simulation.py # Targeted retention intervention simulation
│       ├── verify_phase4_step2.py             # Feature set inclusion & CM verifier
│       ├── test_sim_rules.py                  # Scenario rule overlap tester
│       ├── phase4_step1_driver_analysis_report.md # Step 1 driver analysis report
│       ├── phase4_step2_ablation_report.md    # Step 2 feature ablation report
│       └── phase4_step3_retention_report.md   # Step 3 retention simulation report
│
├── reports/                                   # Assessment deliverables & master reports
│   ├── FINAL_PROJECT_REPORT.md                # Comprehensive 12-section technical & business report
│   ├── assessment/
│   │   ├── Customer_Churn_Prediction_Assessment.docx # Assessment brief
│   │   └── Propel Assessments.docx            # Internship overview
│
├── scripts/                                   # General-purpose verification & utility scripts
│   ├── final_verification_check.py            # Master comprehensive quality gate audit
│   ├── update_and_run_notebook.py             # Notebook runner utility
│   ├── update_notebook_paths.py               # Notebook path updater
│   └── clean_markdown_links.py                # Markdown link cleaner
│
└── tests/                                     # Automated unit & regression test suite
    ├── __init__.py
    ├── test_assessment4_deliverables.py       # Batch prediction, segment cohort & ROI unit tests
    ├── test_prediction.py                     # Model loading, schema & Customer 500014 benchmark test
    └── test_final_verification.py             # Full quality gate audit regression test
```

---

## 4. Environment Setup & Dependencies

The project is developed in **Python 3.10+** (tested version; verified on Python 3.12 and Python 3.13).

### Installation & Virtual Environment Setup
From the project root:
```bash
# 1. Use Python 3.10+ (tested version) and create a fresh virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd):
.venv\Scripts\activate.bat
# On Linux/macOS:
source .venv/bin/activate

# 3. Install strictly pinned dependencies
pip install -r config/requirements.txt
```

`config/requirements.txt` contains pinned exact versions:
```text
# Tested on Python 3.10+ (tested on Python 3.12 and Python 3.13)
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.6.1
shap==0.54.1
openpyxl==3.1.5
joblib==1.4.2
matplotlib==3.10.0
seaborn==0.13.2
scipy==1.14.1
```

---

## 5. Execution & Reproducibility Guide

### Interactive Customer Prediction (Primary Entry Point)
To enter custom customer attributes interactively and predict churn probability in real time from the root directory:
```bash
python predict_customer.py
```

### Automated Benchmark Verification (Customer 500014)
To run automated verification on known benchmark customer `500014` (validates pipeline against verified baseline of **77.67%**):
```bash
python predict_customer.py --verify
```

### Automated Unit & Regression Tests
To run the automated unittest suite (model loading, 28-feature schema, benchmark prediction, quality gates):
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Comprehensive Quality Gate Audit
To verify all 49 canonical deliverables, exact metrics, confusion matrices, ablation sets, scenario counts, and segment totals:
```bash
python scripts/final_verification_check.py
```

### Phase Pipeline Workflows

#### Phase 1: Exploratory Data Analysis
```bash
python phases/phase1/generate_phase1_eda.py
# Output: 7 charts saved to phases/phase1/output/charts/
```

#### Phase 2: Feature Engineering & Dataset Construction
```bash
python phases/phase2/build_phase2_dataset.py
# Output: data/processed/phase2_modeling_dataset.csv and phases/phase2/output/
```

#### Phase 3: Model Training & Evaluation
```bash
python phases/phase3/train_phase3_models.py
# Or end-to-end execution:
python phases/phase3/execute_phase3.py
# Verification:
python phases/phase3/verify_phase3.py
# Threshold Decision Sensitivity Analysis (Training-only 5-fold CV):
python phases/phase3/threshold_comparison.py
# Bootstrap Confidence Intervals (1,000 resamples, seed 42, paired PR-AUC difference test):
python phases/phase3/bootstrap_confidence_intervals.py
# Output: phases/phase3/output/ validation predictions, bootstrap CIs, and metrics CSVs
```

#### Customer-Level Batch Predictions & Local SHAP Attribution Engine
```bash
python phases/phase3/batch_predict.py
# Outputs:
# 1. phases/phase3/output/phase3_batch_predictions_all_eligible.csv (All 2,549 eligible customers)
# 2. phases/phase3/output/phase3_validation_predictions_with_drivers.csv (602 validation customers)
```
* **Storage Location:** `phases/phase3/output/phase3_batch_predictions_all_eligible.csv`
* **Column Definitions:**
  * `CUSTOMER_ID`: Unique integer identifier for each eligible customer ($N = 2,549$; 51 post-cutoff signups excluded).
  * `CHURN_PROBABILITY`: Model-predicted churn likelihood bounded in $[0, 1]$.
  * `PREDICTED_LABEL`: Binary classification $\{0, 1\}$ at the canonical decision threshold $\tau = 0.50$.
  * `TOP_DRIVER_1`: Leading transformed feature driving the prediction, selected by largest absolute SHAP contribution ($|\phi_1|$).
  * `TOP_DRIVER_1_CONTRIBUTION`: Signed SHAP attribution on the **probability scale** ($+0.0635 = +6.35$ percentage points).
  * `TOP_DRIVER_2`: Second leading transformed feature driving the prediction ($|\phi_2|$), distinct from Driver 1.
  * `TOP_DRIVER_2_CONTRIBUTION`: Signed SHAP attribution on the probability scale.
* **SHAP Scale & Interpretation:** Attributions are computed via `shap.TreeExplainer` on the churn probability scale, satisfying exact additive reconstruction: $\hat{p} = E[Y=1] + \sum \phi_i$ from the baseline prior $E[Y=1] = 0.4987$. Driver attributions reflect observational statistical associations, not causal intervention guarantees.

#### Phase 4: Model Interpretability & Driver Analysis
```bash
python phases/phase4/phase4_step1_driver_analysis.py
# Output: phases/phase4/output/ SHAP and segment risk summaries
```

#### Customer Segment & Signup Cohort Churn Risk Analysis
```bash
python phases/phase4/phase4_segment_cohort_analysis.py
# Output:
# 1. phases/phase4/output/phase4_segment_cohort_churn_analysis.csv (Tier, City, Store Price Tier, Cohort)
# 2. phases/phase4/output/phase4_signup_cohort_analysis.csv (Tenure & churn breakdown across 2,600 accounts)
# 3. phases/phase4/output/charts/phase4_signup_cohort_distribution.png
```

#### Phase 4: Feature Ablation Study
```bash
python phases/phase4/phase4_step2_ablation_study.py
# Verification:
python phases/phase4/verify_phase4_step2.py
# Output: phases/phase4/output/ ablation results and comparison chart
```

#### Phase 4: Retention Intervention Simulation
```bash
python phases/phase4/phase4_step3_retention_simulation.py
# Scenario rules validation:
python phases/phase4/test_sim_rules.py
# Output: phases/phase4/output/ targeting roster, scenario metrics, and simulation chart
```

#### Retention Financial Cost, Benefit & ROI Sensitivity Modeling
```bash
python phases/phase4/phase4_retention_roi_analysis.py
# Output:
# 1. phases/phase4/output/phase4_roi_assumptions.csv (Defensible illustrative retail parameters)
# 2. phases/phase4/output/phase4_retention_financial_roi_sensitivity.csv (12-scenario sensitivity matrix)
# 3. phases/phase4/output/charts/phase4_retention_roi_sensitivity.png
```

---

## 6. Phase-by-Phase Deliverables Summary

- **Phase 1: Exploratory Data Analysis & Schema Audit**
  - Audited 5 relational sheets, resolving missing categorical values and transaction anomalies.
  - Verified 100% referential integrity across customer foreign keys.
- **Phase 2: Temporal Architecture & Feature Engineering**
  - Established March 31, 2024 temporal boundary to eradicate lookahead leakage.
  - Engineered 28 pre-cutoff predictors (24 numerical, 4 categorical) spanning transactions, support tickets, mobile app, and email.
- **Phase 3: Baseline Development & Model Validation**
  - Benchmarked Logistic Regression against Random Forest on the validation cohort ($N = 602$).
  - Identified the operational trade-off: LR precision (89.55%) vs. RF recall (82.35%).
  - **Customer-Level Batch Predictions & Local SHAP Attribution:** Evaluated all 2,549 eligible loyalty members (`phase3_batch_predictions_all_eligible.csv`), generating individual churn probabilities, binary classifications ($\tau = 0.50$), and local Top-2 SHAP drivers. Confirmed pipeline benchmark on customer `500014` at **77.67%**.
- **Phase 4: Advanced Synthesis, Ablation & Operational Simulation**
  - **Step 1: Driver Analysis & Segment Profiling:** SHAP attribution establishes `TXNS_LAST_3M`, `RECENCY_MONTHS_PRE_CUTOFF`, and `SPEND_LAST_3M` as dominant predictive drivers. Segment audit reconciles all 602 validation members across tiers (Silver: 348, Gold: 191, Platinum: 63) and store price tiers (Mainstream: 281, Value: 198, Upscale: 100, Unknown: 23).
  - **Customer Segment & Signup Cohort Churn Risk Analysis:** Reconciled risk across Membership Tiers, Cities, Home Store Price Tiers, and Signup Cohorts (`phase4_segment_cohort_churn_analysis.csv` and `phase4_signup_cohort_analysis.csv`). Identified significant cohort tenure shift: mature signups (2021–2022) exhibit ~37%–38% churn, whereas recent joiners (2024 Q1) exhibit only 2.04% churn due to short pre-cutoff exposure.
  - **Step 2: Feature Ablation Study:** Evaluated 4 cumulative feature sets:
    | Feature Set | Features | Accuracy | Precision | Recall | PR-AUC | ROC-AUC | TP / FP |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | Set 1: Demographics Only | 6 | 0.5482 | 0.1390 | 0.4235 | 0.1410 | 0.4807 | 36 / 223 |
    | Set 2: Demo + Tenure/History | 10 | 0.1495 | 0.1400 | 0.9765 | 0.1791 | 0.5915 | 83 / 510 |
    | Set 3: Demo + Tenure + Purchases | 22 | 0.9103 | 0.6598 | 0.7529 | 0.8290 | 0.9327 | 64 / 33 |
    | Set 4: Full Set (+ Support & Digital) | 28 | 0.9169 | 0.6667 | 0.8235 | 0.8499 | 0.9446 | 70 / 35 |
  - **Step 3: Retention Intervention Simulation (105 Flagged At-Risk Members):**
    | Scenario | Targeting Rule | Targeted ($N$) | Churners Captured | False Alarms ($FP$) | Precision (%) | Hypothetical Saved (@10% / 20% / 30%)* |
    | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
    | **S1: Support Follow-up** | Tickets $>0$ OR Complaint $=1$ | 54 | 45 | 9 | 83.3% | 4.5 / 9.0 / 13.5 |
    | **S2: Reactivation Offer** | Recency $\ge 2$ calendar months | 55 | 52 | 3 | 94.5% | 5.2 / 10.4 / 15.6 |
    | **S3: Digital Re-engagement**| Monthly App $=0$ OR Email $=0$ | 51 | 35 | 16 | 68.6% | 3.5 / 7.0 / 10.5 |
    | **Combined (Deduplicated)** | **S1 OR S2 OR S3 (Unique)** | **91** | **67** | **24** | **73.6%** | **6.7 / 13.4 / 20.1** |
    | **Untargeted Members** | No scenario rule matched | 14 | 3 | 11 | 21.4% | N/A (General Drift) |
  - **Retention Financial Cost, Benefit & ROI Sensitivity Modeling:**
    Constructed an illustrative business model (`phase4_retention_financial_roi_sensitivity.csv`) using defensible retail grocery benchmarks (annual customer contribution margin LTV = $250):
    | Scenario | Unit Cost | Total Cost ($) | Efficacy (10% / 20% / 30%) | Expected Saved | Net Benefit ($) | Expected ROI (%) |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
    | **S1: Support Outreach** | $15.00 | $810.00 | 10% / 20% / 30% | 4.5 / 9.0 / 13.5 | +$315 / +$1,440 / +$2,565 | 38.9% / 177.8% / 316.7% |
    | **S2: Reactivation Vouchers**| $8.00 | $440.00 | 10% / 20% / 30% | 5.2 / 10.4 / 15.6 | +$860 / +$2,160 / +$3,460 | 195.5% / 490.9% / 786.4% |
    | **S3: Digital Re-engagement**| $2.00 | $102.00 | 10% / 20% / 30% | 3.5 / 7.0 / 10.5 | +$773 / +$1,648 / +$2,523 | 757.8% / 1615.7% / 2473.5% |
    | **Combined Program** | $10.00 | $910.00 | 10% / 20% / 30% | 6.7 / 13.4 / 20.1 | +$765 / +$2,440 / +$4,115 | 84.1% / 268.1% / 452.2% |
    *\*Note: Financial figures and saved member volumes are illustrative sensitivity models for capacity planning, not measured historical outcomes.*

---

## 7. Project Report

The complete technical and business documentation is available in
[FINAL_PROJECT_REPORT.md](reports/FINAL_PROJECT_REPORT.md).

---

## 8. Project Presentation & Google Slides Deliverable

### Google Slides Executive Presentation (10 Slides)
- **Live Presentation Link:** [FreshBasket Customer Churn Prediction & Proactive Retention Engine (Google Slides)](https://docs.google.com/presentation/d/1pI8quuO_O1MC8b6-e9jm4_fHYE3un5Dh/edit?usp=sharing&ouid=105329998739422544837&rtpof=true&sd=true)
- **Canonical Presentation File:** [`reports/presentation/presentation.pptx`](reports/presentation/presentation.pptx)
- **Complete Speaker Scripts & Timed Notes:** [`reports/presentation/FINAL_PRESENTATION_SLIDES.md`](reports/presentation/FINAL_PRESENTATION_SLIDES.md)

#### 10-Slide Structure & Presentation Delivery Map (10–15 Minutes)

The executive presentation is structured into exactly 10 concise, high-impact slides fully aligned with the Tailwyndz Propel Assessment 4 criteria:

| Slide # | Slide Title | Key Content & Assessment Focus | Delivery Target |
| :---: | :--- | :--- | :---: |
| **Slide 1** | Title & Executive Overview | Business Goal, Technology Stack (Python, Scikit-learn, SHAP), Presenter & Academic Affiliation | 1:00 min |
| **Slide 2** | Business Problem & Objectives | Rising Churn, Untargeted Promotions, Dual Objectives (Risk Prediction & Churn Diagnosis) | 1:15 min |
| **Slide 3** | Dataset & Data Quality | 3 Relational Tables (2,600 Members, 18 Months Activity), Cleaning (8 Profile, 5 Churn, 60 Activity Rows Dropped), Deduplication | 1:30 min |
| **Slide 4** | Churn Definition & Leakage Prevention | Strict March 31, 2024 Cutoff, Q2 Churn Window, Zero Leakage, Cohort Splits (Train: 1,947, Val: 602, Excluded: 51) | 1:30 min |
| **Slide 5** | Feature Engineering & Model Development | 28 Leakage-Safe Predictors (4 Domains), Logistic Regression Baseline vs. 200-Tree Random Forest Ensemble | 1:15 min |
| **Slide 6** | Model Evaluation & Comparison | Validation Metrics Table ($\tau = 0.50$), Precision (89.55% vs. 66.67%), Recall (70.59% vs. 82.35%), Paired Bootstrap CI | 2:00 min |
| **Slide 7** | Churn Drivers & SHAP Explainability | Gini Importance (Transactions, Recency, Spend), TreeSHAP on Probability Scale, Customer 500014 Case Study | 1:30 min |
| **Slide 8** | Feature Ablation & Customer Segmentation | 4-Tier Ablation Study (PR-AUC 0.1410 to 0.8499), Subgroup Churn Rates (Silver, Gold, Platinum Small Sample Caveat) | 1:30 min |
| **Slide 9** | Model Limitations & Challenges | 85 Actual Validation Churners, Probability Calibration Shift (LR 13.86% vs. RF 30.36%), Simulated ROI Nature | 1:15 min |
| **Slide 10** | Conclusion & Retention Recommendations | 3 Operational Playbooks (Support, Reactivation, Digital), Strategic Deployment Roadmap, Projected Savings | 1:30 min |

---

## 9. Troubleshooting Guide

### Missing Model Artifact or File Not Found
- Ensure `models/phase3_random_forest_pipeline.joblib` exists. The project uses dynamic path resolution via `phases/paths.py`, so all scripts automatically find files relative to the project root.

### Running from Subdirectories
- Scripts and tests are working-directory independent. Whether launched from the root `AADHI/` or a subfolder like `phases/phase3/` or `scripts/`, `phases/paths.py` dynamically anchors paths to the project root.

### Notebooks in Jupyter vs. Root
- All four notebooks (`phases/phase1/phase1_cleaning_notebook.ipynb`, `phases/phase1/phase1_eda_and_data_quality_notebook.ipynb`, `phases/phase2/phase2_feature_engineering_notebook.ipynb`, `phases/phase3/phase3_model_development_notebook.ipynb`) reside directly inside their respective phase folder with adaptive path resolution that detects the project root.

---
*Developed for Tailwyndz Technologies Propel Internship Assessment 4 — Retail Customer Churn Prediction.*
