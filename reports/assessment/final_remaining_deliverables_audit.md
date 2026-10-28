# AADHI — Assessment 4 Targeted Audit: Segment Analysis, SHAP Batch Predictions, and EDA Notebooks

**Document Path:** `reports/assessment/final_remaining_deliverables_audit.md`  
**Execution Timestamp:** 2026-10-27T14:35:30+05:30  
**Role:** Senior Data Scientist & Independent Code Auditor  
**Audit Scope:** Rigorous, independent inspection and targeted remediation of the three core Assessment 4 deliverables:
1. Churn comparison by membership tier, city, and signup cohort.
2. Prediction file for all eligible customers with top-two SHAP drivers.
3. EDA notebook with saved, successfully executed outputs.

---

## 1. Tier, City, and Signup-Cohort Audit

### 1.1 File Paths Inspected
- `phases/phase4/phase4_segment_cohort_analysis.py`
- `phases/phase4/output/phase4_segment_cohort_churn_analysis.csv`
- `phases/phase4/output/phase4_signup_cohort_analysis.csv`
- `phases/phase4/output/charts/phase4_signup_cohort_distribution.png`
- `reports/FINAL_PROJECT_REPORT.md` (Section 8: Customer Segment Risk Profiles & Reliability)
- `scripts/final_verification_check.py`

### 1.2 Recomputed Counts and Rates
Every single customer segment and cohort breakdown was recomputed directly from the underlying processed modeling dataset (`data/processed/phase2_modeling_dataset.csv`).

Every row strictly satisfies:
$$\text{Customer\_Count} = \text{Churner\_Count} + \text{Retained\_Count}$$
$$\text{Observed\_Churn\_Rate} = \frac{\text{Churner\_Count}}{\text{Customer\_Count}} \times 100\%$$

#### A. Membership Tier (Validation Cohort $N = 602$, 100% Reconciled)
- **Silver:** Customer Count = 348, Churner Count = 45, Retained Count = 303, Observed Churn Rate = **12.93%**
- **Gold:** Customer Count = 191, Churner Count = 26, Retained Count = 165, Observed Churn Rate = **13.61%**
- **Platinum:** Customer Count = 63, Churner Count = 14, Retained Count = 49, Observed Churn Rate = **22.22%**
- **Reconciliation Total:** Total Customers = 602, Total Churners = 85, Total Retained = 517. Zero missing or dropped tiers.

#### B. City Breakdown (Validation Cohort $N = 602$, 10 Municipal Markets)
- **Austin:** Count = 64, Churners = 13, Retained = 51, Rate = **20.31%**
- **Boise:** Count = 60, Churners = 7, Retained = 53, Rate = **11.67%**
- **Columbus:** Count = 58, Churners = 9, Retained = 49, Rate = **15.52%**
- **Denver:** Count = 63, Churners = 10, Retained = 53, Rate = **15.87%**
- **Madison:** Count = 62, Churners = 10, Retained = 52, Rate = **16.13%**
- **Nashville:** Count = 43, Churners = 5, Retained = 38, Rate = **11.63%** (Smallest sample)
- **Portland:** Count = 70, Churners = 8, Retained = 62, Rate = **11.43%**
- **Raleigh:** Count = 61, Churners = 6, Retained = 55, Rate = **9.84%** (Lowest rate)
- **Spokane:** Count = 50, Churners = 6, Retained = 44, Rate = **12.00%**
- **Tulsa:** Count = 71, Churners = 11, Retained = 60, Rate = **15.49%** (Largest market)
- **Reconciliation Total:** Total Customers = 602, Total Churners = 85, Total Retained = 517. Zero nulls.
- **Statistical Reliability:** Validation samples per city range from 43 to 71, with only 5 to 13 churners each. Binomial confidence intervals overlap substantially (e.g., Austin $[11.3\%, 32.2\%]$ vs. Raleigh $[3.7\%, 20.2\%]$). City variance is not statistically significant; geography provides negligible predictive power compared to transactional recency and customer service friction.

#### C. Signup Cohort Dynamics (Full Program Database $N = 2,600$, 100% Reconciled)
- **2021 Signups (Early Adopters):** Count = 745, Churners = 279, Retained = 466, Rate = **37.45%** (Train Split)
- **2022 Signups (Mature Base):** Count = 792, Churners = 306, Retained = 486, Rate = **38.64%** (Train Split)
- **2023 H1 Signups (Pre-Cutoff Train):** Count = 410, Churners = 148, Retained = 262, Rate = **36.10%** (Train Split)
- **2023 Q3 Signups (Val: 9–12M Tenure):** Count = 202, Churners = 51, Retained = 151, Rate = **25.25%** (Val Split)
- **2023 Q4 Signups (Val: 6–9M Tenure):** Count = 204, Churners = 30, Retained = 174, Rate = **14.71%** (Val Split)
- **2024 Q1 Signups (Val: 0–3M Tenure):** Count = 196, Churners = 4, Retained = 192, Rate = **2.04%** (Val Split)
- **2024 Q2 Signups (Post-Cutoff Excluded):** Count = 51, Churners = 0, Retained = 51, Rate = **0.00%** (Excluded)
- **Reconciliation Total:** Total Customers = 2,600, Total Churners = 818, Total Retained = 1,782.
- **Exposure Guardrail:** Members enrolled in Q1 2024 joined 0 to 3 months before the March 31 cutoff. Because onboarding transactions are intrinsically recent, virtually none could qualify as churners under the 90-day post-cutoff inactivity definition. This is an artifact of exposure duration and tenure, not a causal retention effect.

### 1.3 Missing Comparisons & Corrections Made
1. **Defect:** `phase4_segment_cohort_churn_analysis.csv` previously contained City and Home Store Price Tier only for `Validation_Cohort_N602`, but lacked macro breakdowns for `All_Eligible_N2549` and `Full_Population_N2600`.
2. **Correction:** Updated `phases/phase4/phase4_segment_cohort_analysis.py` to record all 4 dimensions (Membership Tier, City, Store Tier, Signup Cohort) across all three population scopes (`Validation_Cohort_N602`, `All_Eligible_N2549`, `Full_Population_N2600`), expanding the table to 67 fully reconciled rows.
3. **Report Integration:** Updated `reports/FINAL_PROJECT_REPORT.md` Section 8.3 with the complete 10-city table, binomial confidence intervals, and explicit sample reliability warnings.

---

## 2. Batch Prediction and Top Two SHAP Drivers Audit

### 2.1 File Paths Inspected
- `phases/phase3/batch_predict.py`
- `phases/phase3/output/phase3_batch_predictions_all_eligible.csv`
- `phases/phase3/output/phase3_validation_predictions_with_drivers.csv`
- `models/phase3_random_forest_pipeline.joblib`
- `data/processed/phase2_modeling_dataset.csv`

### 2.2 Population Coverage & ID Integrity
- **Total Master Database:** `2,600` accounts.
- **Ineligible Post-Cutoff Signups:** Exactly `51` accounts enrolled in April–May 2024 with zero pre-cutoff observation months; held out to prevent lookahead leakage.
- **Expected Eligible Population:** `2,549` accounts ($2,600 - 51$).
- **Actual Batch Predictions Count:** Exactly `2,549` rows in `phase3_batch_predictions_all_eligible.csv`.
- **Unique Non-Null IDs:** Exactly 2,549 unique customer IDs, 0 nulls, 0 duplicate IDs, 0 post-cutoff signups included.
- **Validation Subset:** Exactly `602` rows in `phase3_validation_predictions_with_drivers.csv`, capturing 85 observed churners and 105 predicted positive accounts at threshold $\tau = 0.50$.

### 2.3 Required Columns & Probability Integrity
- **Mandatory Columns Verified:**
  - `CUSTOMER_ID`: Integer, non-null, unique.
  - `CHURN_PROBABILITY`: Float, strictly bounded in $[0.0046, 0.9978]$.
  - `PREDICTED_LABEL`: Binary integer $\{0, 1\}$, strictly matching $(p \ge 0.50)$.
  - `TOP_DRIVER_1`: String, valid preprocessor transformed feature name.
  - `TOP_DRIVER_1_CONTRIBUTION`: Float, non-zero signed contribution.
  - `TOP_DRIVER_2`: String, valid preprocessor transformed feature name.
  - `TOP_DRIVER_2_CONTRIBUTION`: Float, non-zero signed contribution.
- **Distinct Drivers:** Verified that for 100% of the 2,549 accounts, `TOP_DRIVER_1 != TOP_DRIVER_2`. Zero duplicate driver pairs.
- **No Administrative Leakage:** Zero IDs, target labels, or administrative columns appear in driver fields.

### 2.4 SHAP Scale & Mathematical Additive Reconstruction
- **Explainer:** `shap.TreeExplainer` applied directly to `RandomForestClassifier`.
- **Output Scale:** Direct probability scale $[0, 1]$ (not raw log-odds margins).
- **Baseline Prior:** $E[Y=1] = 0.498732$ (~$49.87\%$ baseline training churn expectation).
- **Exact Additivity Verification:** For any customer, $\hat{p} = E[Y=1] + \sum_{j=1}^{28} \phi_j$.
- **Reconstruction Precision:** Recomputed on sample customers; maximum numerical divergence between pipeline `predict_proba` and SHAP reconstruction was $\le 2.8 \times 10^{-15}$.

### 2.5 Customer 500014 Benchmark Verification
- **Target Customer ID:** `500014`
- **Observed Target Label:** `1` (Actual Churner in Q2 2024)
- **Saved Prediction:** `77.67%` ($0.7767$)
- **Live Fresh Pipeline Prediction:** `77.67%` ($0.776706$)
- **Predicted Class:** `1` (at threshold $\tau = 0.50$)
- **Top Driver 1:** `SPEND_LAST_3M` (Attribution = **$+0.063462$**, or $+6.35$ percentage points). Customer spent only $30.00 across the preceding 90 days.
- **Top Driver 2:** `TXNS_LAST_3M` (Attribution = **$+0.058745$**, or $+5.87$ percentage points). Customer visited only once in Q1 2024.
- **Model Invariance:** `batch_predict.py` strictly loaded the pipeline in read-only inference mode; zero model retraining occurred.

---

## 3. EDA Notebook Audit

### 3.1 Notebook Paths & Canonical Designation
- **Canonical Comprehensive EDA Notebook:** `phases/phase1/phase1_eda_and_data_quality_notebook.ipynb`
- **Convenience Root Mirror:** `notebooks/phase1_eda_and_data_quality_notebook.ipynb` (byte-for-byte identical mirror)
- **Phase 1 Data Cleaning Notebook:** `phases/phase1/phase1_cleaning_notebook.ipynb`
- **Cleaning Mirror:** `notebooks/phase1_cleaning_notebook.ipynb` (byte-for-byte identical mirror)

### 3.2 Cell Structure & Execution Evidence

| Notebook | Total Cells | Code Cells | Cells with Saved Outputs | Error Outputs | Execution Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `phase1_eda_and_data_quality_notebook.ipynb` | 23 | 11 | 11 (100%) | **0** | **Successfully Executed** |
| `notebooks/phase1_eda_and_data_quality_notebook.ipynb` | 23 | 11 | 11 (100%) | **0** | **Successfully Executed** |
| `phases/phase1/phase1_cleaning_notebook.ipynb` | 15 | 7 | 7 (100%) | **0** | **Successfully Executed** |
| `notebooks/phase1_cleaning_notebook.ipynb` | 15 | 7 | 7 (100%) | **0** | **Successfully Executed** |

### 3.3 Required Data Quality Topics Visibly Documented
1. **Duplicate Rows:** Documents 8 duplicates in `fb Customer Profile`, 5 in `fb Churn Label`, 58 in `fb Monthly Activity`, and 120 rows involved in `(CUSTOMER_ID, MONTH)` key collisions.
2. **Missing Value Patterns:** Ingests and documents missing data across tables: `HOME_STORE_PRICE_TIER` (130 missing, 4.98%), `AGE` (78 missing, 2.99%), `GENDER` (52 missing, 1.99%), and `LAST_PURCHASE_DATE` (120 missing, 4.61%).
3. **Negative & Corrupted Spend:** Flags exactly 24 negative spend rows and 20 zero-spend rows with positive transaction counts.
4. **Text Casing & Inconsistencies:** Demonstrates title-casing and whitespace trimming for `CITY` and `MEMBERSHIP_TIER`.
5. **Missing Months & Incomplete Histories:** Documents and flags the 51 post-cutoff signups who enrolled after the March 31 cutoff.
6. **Ground Truth Churn Target:** Proves the official churn definition: exactly zero transactions in the April–June 2024 outcome window despite pre-cutoff loyalty enrollment.

### 3.4 Corrections Made During Audit
1. **Defect:** `phases/phase1/phase1_cleaning_notebook.ipynb` previously had unexecuted code cells (`outputs: []`) and was missing `import pandas as pd` in Cell 1.
2. **Correction:** Injected `import pandas as pd` and root-safe export paths into `phase1_cleaning_notebook.ipynb`, executed the notebook from start to finish using `nbconvert`, verified that all 7 code cells generated valid outputs with 0 errors, and synchronized the executed notebook to `notebooks/phase1_cleaning_notebook.ipynb`.

---

## 4. Regression Verification & Commands Executed

Every test suite, compiler check, and verification script was executed directly from the project root:

| Command | Exit Code | Stdout Summary | Status |
| :--- | :---: | :--- | :---: |
| `powershell -Command "Get-ChildItem -File"` | `0` | Strictly `predict_customer.py` and `README.md` at root | **PASS** |
| `python -m compileall .` | `0` | Listing and compiling all project packages; 0 syntax errors | **PASS** |
| `python predict_customer.py --verify` | `0` | Customer 500014 live probability matches 77.67% | **PASS** |
| `python -m pytest tests/` | `0` | 11 passed in 10.90s (all unit & integration tests) | **PASS** |
| `python scripts/final_verification_check.py` | `0` | All 63 canonical deliverables, mirrors, metrics, & reconciliations verified | **PASS** |
| `python phases/phase4/phase4_segment_cohort_analysis.py` | `0` | Generated 67-row reconciled segment/cohort table & chart | **PASS** |

### Remaining Practical Limitations
- **Illustrative Financial Assumptions:** Campaign intervention unit costs ($15 phone outreach, $25 reactivation offer, $5 digital nudge) and the $250 Annual Contribution Margin are planning assumptions for sensitivity analysis. FreshBasket should update these with audited ERP ledger values before live budgeting.
- **Q1 2024 Cohort Truncation:** Observed churn in Q1 2024 is 2.04% because customers joined 0–3 months prior to the cutoff. This represents tenure exposure truncation, not a proven causal retention effect.

---

## 5. Final Verdict

### **PASS**

**Reasoning:**
1. **Tier, City, & Cohort Analysis:** All three dimensions are comprehensively calculated, independently reconciled across all relevant scopes (Validation $N=602$, Eligible $N=2,549$, Full $N=2,600$), documented with sample reliability caveats in `reports/FINAL_PROJECT_REPORT.md`, and validated by automated tests.
2. **Batch Prediction & SHAP:** All 2,549 eligible accounts are scored once without duplicates or ineligible signups. Probabilities are strictly bounded in $[0, 1]$ matching threshold 0.50. Top-two drivers are 100% distinct, valid transformed features. SHAP values operate on the exact additive probability scale with reconstruction error $< 2.8 \times 10^{-15}$. Customer 500014 benchmark is 77.67%.
3. **EDA Notebooks:** Both the comprehensive EDA notebook and the cleaning notebook have 100% executed code cells, saved embedded outputs, and zero error outputs. All assessment data quality topics are visibly analyzed.
4. **Safety & Hygiene:** Root directory contains strictly `predict_customer.py` and `README.md`. No model retraining, feature schema modification, or cutoff tampering occurred. External backups exist outside the active project tree.
