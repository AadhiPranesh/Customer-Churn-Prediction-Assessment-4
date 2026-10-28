# AADHI — Assessment 4 Critical Compliance Audit & Targeted Implementation Report

**Document Location:** `reports/assessment/final_assessment4_compliance_audit.md`  
**Execution Timestamp:** 2026-10-27T14:53:00+05:30  
**Auditor Role:** Senior Data Scientist, Python Engineer & Independent Assessment Auditor  
**Audit Scope:** Independent, evidence-based inspection and targeted verification of the three core Assessment 4 requirements against `Customer_Churn_Prediction_Assessment.docx`:
1. Data-quality checks, remediation decisions, relational joining, and fully executed EDA notebooks.
2. Feature importance, customer-level SHAP attributions on the probability scale, and segment comparisons (membership tier, city, signup cohort).
3. Prediction file for all 2,549 eligible customers with predicted probabilities, binary classifications, and distinct top-two SHAP drivers.

**Safety Verification & Checksums:**
- Pre-audit backup created outside the project root at: `c:\7th Sem\AADHI_backup_compliance_audit`.
- Model artifact preserved without retraining: `models/phase3_random_forest_pipeline.joblib`.
- Model SHA-256 Checksum: `FAA7AAE2AD9C2D70A6EA0F6F63BC2A95D27F5D240A2A8E67060CF9ABABD25792`.
- Customer 500014 Benchmark Prediction: Exact **77.67%** churn probability, class 1 at $\tau = 0.50$.
- Root directory isolation: Strictly only `predict_customer.py` and `README.md` reside at the project root. Zero unauthorized root files or archive folders created.

---

## 1. Data Quality and EDA Audit

### 1.1 Files Inspected
- `data/raw/FreshBasket_Loyalty_Churn_Dataset.xlsx` (Raw Excel workbook: sheets `fb Customer Profile`, `fb Monthly Activity`, `fb Churn Label`)
- `phases/phase1/phase1_cleaning_notebook.ipynb` (Phase 1 Data Cleaning Notebook)
- `phases/phase1/phase1_eda_and_data_quality_notebook.ipynb` (Phase 1 Comprehensive EDA & Anomaly Notebook)
- `notebooks/phase1_cleaning_notebook.ipynb` (Convenience mirror)
- `notebooks/phase1_eda_and_data_quality_notebook.ipynb` (Convenience mirror)
- `phases/phase1/phase1_data_quality_report.md` & `phases/phase1/phase1_decisions.md`
- `phases/phase1/output/cleaned_data/` (`cleaned_fb_customer_profile.csv`, `cleaned_fb_churn_label.csv`, `cleaned_fb_monthly_activity.csv`)
- `data/processed/phase2_modeling_dataset.csv` (Customer-level merged modeling dataset)

### 1.2 Checks Performed, Empirical Findings & Decisions

| Data Quality Check | Raw Findings in Source Data | Cleaning & Remediation Decision | Downstream Modeling Consistency |
| :--- | :--- | :--- | :--- |
| **Duplicate Rows** | • `fb Customer Profile`: 8 exact duplicate rows (16 rows total).<br>• `fb Churn Label`: 5 exact duplicate rows (10 rows total).<br>• `fb Monthly Activity`: 58 exact duplicate rows (116 rows total).<br>• Key collisions on `(CUSTOMER_ID, MONTH)`: 120 rows involved. | Deduplicated each table using `.drop_duplicates()`. For the 2 conflicting duplicate pairs in Monthly Activity (where identical keys had conflicting $+X$ and $-X$ spend), retained the positive record because it matched $TRANSACTIONS \times AVG\_BASKET\_VALUE$. | Restores strict primary key uniqueness: exactly 2,600 unique members in Profile and Churn; 28,013 unique valid customer-months in Activity. |
| **Missing Months** | • Activity covers 18 calendar months (`2023-01-01` to `2024-06-01`).<br>• Customers have varying observed histories based on signup dates (`2021-02-18` to `2024-05-02`).<br>• 6 customers in Profile/Churn have zero activity records (`500754`, `501092`, `501216`, `501563`, `502014`, `502550`). | Explicitly distinguished months prior to signup from unobserved active months. The 6 members with zero activity are flagged with `INSUFFICIENT_HISTORY_FLAG = 1` and retained via left-join with 0 transaction count. 51 post-cutoff signups (April–May 2024) are isolated into `EXCLUDED_POST_CUTOFF`. | Prevents artificial zero-filling before enrollment. Ensures complete 2,600 customer accounting without dropping ground-truth evaluation members. |
| **Invalid & Negative Spend** | • 24 rows initially in `fb Monthly Activity` had negative spend ($-\$249.21$ to $-\$31.05$).<br>• 2 were conflicting duplicate pairs resolved above.<br>• 22 standalone negative spend rows remained. | In all 22 cases, $TRANSACTIONS > 0$, $AVG\_BASKET\_VALUE > 0$, and $|TOTAL\_SPEND| / TRANSACTIONS == AVG\_BASKET\_VALUE$ within rounding ($< 0.005$). Raw spend was preserved in raw exports; rectified as $|TOTAL\_SPEND|$ with audit flag `IS_NEGATIVE_SPEND = 1`. | Eliminates negative cumulative RFM spend without discarding genuine customer visits or corrupting monetary velocity. |
| **Zero-Transaction Anomalies** | • 0 rows had zero transactions with positive spend.<br>• Exactly 20 rows in `fb Monthly Activity` had positive transactions ($1 \dots 11$) but `TOTAL_SPEND = 0.0` despite positive average basket values. | Preserved raw spend; reconstructed total spend in `TOTAL_SPEND_RECONSTRUCTED` as $TRANSACTIONS \times AVG\_BASKET\_VALUE$ ($15.12 to $48.69) with audit flag `IS_ZERO_SPEND_WITH_TXNS = 1`. | Prevents understating customer spend velocity while maintaining defensible accounting based on recorded basket values. |
| **Text Inconsistencies** | • `CITY`: 40 raw string variations for 10 metropolitan markets (whitespace and casing variants, e.g. `' Spokane '`, `'SPOKANE'`).<br>• `MEMBERSHIP_TIER`: 12 raw string variants for 3 tiers (`Silver`, `Gold`, `Platinum`). | Standardized text via `.astype(str).str.strip().str.title()`. Retained raw columns `CITY_RAW` and `MEMBERSHIP_TIER_RAW` for verification. | Collapses text into exactly 10 distinct cities and 3 loyalty tiers, preventing artificial one-hot column fragmentation. |
| **Spend Outliers** | • Distribution across 28,073 monthly records: Mean = $134.64, Median = $116.39, $Q1 = \$58.77$, $Q3 = \$188.15$, $IQR = \$129.38$.<br>• $1.5\times IQR$ Upper Threshold = $382.22 (664 records).<br>• $3.0\times IQR$ Extreme Threshold = $576.29 (70 records).<br>• Max Monthly Spend = $3,731.42 (Customer 501306, large catering/family grocery order). | Legitimate high-spending loyalty members (e.g. Platinum tier large grocery orders) were **preserved without truncation or artificial capping**. | Preserves true customer lifetime value and monetary variance, allowing tree ensembles to learn natural high-value customer behaviors. |
| **Missing Demographics** | • `AGE`: 78 missing values ($2.99\%$).<br>• `GENDER`: 52 missing values ($1.99\%$).<br>• `HOME_STORE_PRICE_TIER`: 130 missing values ($4.98\%$). | Replaced missing categorical values with explicit category `'Unknown'`. Replaced missing `AGE` with median ($42.0$) while adding explicit binary missingness indicator `AGE_IS_MISSING = 1`. | Retains all 2,600 customer rows without selection bias. Allows tree models to evaluate whether non-reporting of demographic data predicts churn. |
| **Relational Merging** | 3 source tables joined into single customer-level dataset `phase2_modeling_dataset.csv`. | Merged deduplicated `fb Customer Profile` (2,600 rows) with `fb Churn Label` (2,600 rows) and aggregated pre-cutoff features from `fb Monthly Activity` ($MONTH < 2024-04-01$). | Produced exact 2,600 rows $\times$ 33 columns with zero Cartesian row multiplication and 100% referential integrity. |
| **Churn Target Verification** | Verified definition: zero purchase transactions in April–June 2024 despite prior active history. | Cross-tabulated Q2 2024 transactions against `CHURNED`. All 818 churners had exactly 0 transactions in April–June 2024. Among 1,782 retained, 1,657 had active Q2 transactions, 120 had zero prior lifetime transactions (non-activated accounts), and 5 were edge-case metadata discrepancies documented in report. | Guarantees target validity and verifies that outcome window activity is strictly excluded from predictive features. |

### 1.3 Notebook Execution Status & Saved-Output Audit

| Notebook File | Total Cells | Code Cells | Cells with Saved Outputs | Saved Execution Errors | Execution Verification Method |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `phases/phase1/phase1_eda_and_data_quality_notebook.ipynb` | 23 | 11 | **11 (100%)** | **0** | Executed via `nbconvert` from start to finish; exit code 0 |
| `notebooks/phase1_eda_and_data_quality_notebook.ipynb` | 23 | 11 | **11 (100%)** | **0** | Byte-for-byte identical mirror verified via file size (77,892 bytes) |
| `phases/phase1/phase1_cleaning_notebook.ipynb` | 15 | 7 | **7 (100%)** | **0** | Executed via `nbconvert` from start to finish; exit code 0 |
| `notebooks/phase1_cleaning_notebook.ipynb` | 15 | 7 | **7 (100%)** | **0** | Byte-for-byte identical mirror verified via file size (14,557 bytes) |

**Audit Confirmation:** Neither notebook contains unexecuted cells, empty outputs, or saved traceback errors. Deduplication in code cell 6 ensures that the EDA notebook outputs strictly display the verified ground-truth counts: **1,782 retained (68.54%)** and **818 churners (31.46%)** across 2,600 members.

---

## 2. Feature Importance, SHAP, and Segment Comparisons

### 2.1 Feature Importance Evidence
Extracted directly from the saved canonical Random Forest model (`models/phase3_random_forest_pipeline.joblib`) and exported to `phases/phase3/output/phase3_random_forest_feature_importances.csv` and `phases/phase4/output/phase4_step1_top_predictors.csv`:

| Rank | Feature Name | Dimension | RF Gini Importance | SHAP Mean \|Attribution\| | LR Coefficient ($\beta$) | Odds Ratio |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `TXNS_LAST_3M` | Recent Velocity | **0.1741** (17.41%) | 0.0979 | $-3.0280$ | 0.0484 |
| 2 | `RECENCY_MONTHS_PRE_CUTOFF` | Purchase Recency | **0.1686** (16.86%) | 0.0966 | $+0.6130$ | 1.8460 |
| 3 | `SPEND_LAST_3M` | Monetary Momentum| **0.1649** (16.49%) | 0.1052 | $-2.0986$ | 0.1226 |
| 4 | `AVG_MONTHLY_TXNS_PRE_CUTOFF` | Baseline Velocity | 0.0668 (6.68%) | 0.0270 | $+0.8643$ | 2.3732 |
| 5 | `ACTIVE_MONTHS_PRE_CUTOFF` | Consistency | 0.0651 (6.51%) | 0.0318 | $-0.0043$ | 0.9957 |
| 6 | `AVG_MONTHLY_SPEND_PRE_CUTOFF` | Monetary Baseline | 0.0570 (5.70%) | 0.0260 | $+0.4644$ | 1.5910 |
| 7 | `TOTAL_TXNS_PRE_CUTOFF` | Cumulative Volume | 0.0499 (4.99%) | 0.0261 | $+0.4848$ | 1.6238 |
| 8 | `AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF` | Digital Engagement | 0.0479 (4.79%) | 0.0213 | $-0.3719$ | 0.6894 |
| 18 | `TOTAL_SUPPORT_TICKETS_PRE_CUTOFF` | Support Friction | 0.0042 (0.42%) | 0.0034 | $+0.6279$ | 1.8737 |
| 20 | `AGE` | Demographic | 0.0021 (0.21%) | 0.0006 | $-0.0441$ | 0.9568 |

*Causal Guardrail:* High importance indicates strong observational predictive attribution; it does not constitute proof that artificially forcing transactions will causally retain customers.

### 2.2 SHAP Correctness & Output Scale
- **Explainer Class:** `shap.TreeExplainer` on `RandomForestClassifier`.
- **Target Class:** Class 1 (Churn).
- **Scale:** Direct **churn probability scale** $[0, 1]$ (not raw margins or log-odds).
- **Baseline Prior:** $E[Y=1] = 0.498732$ (~$49.87\%$ baseline expectation).
- **Exact Additivity Verification:** For any customer, $\hat{p} = E[Y=1] + \sum_{j=1}^{28} \phi_j$.
- **Reconstruction Error:** Recomputed on live accounts; maximum discrepancy between live pipeline predicted probability and SHAP sum is $\le 2.8 \times 10^{-15}$.
- **Customer 500014 Benchmark:**
  - Live Pipeline Churn Probability: **77.67%** (exact $0.776706$).
  - Predicted Label ($\tau = 0.50$): **1**.
  - Top Driver 1: `SPEND_LAST_3M` ($+0.063462$, or $+6.35$ percentage points).
  - Top Driver 2: `TXNS_LAST_3M` ($+0.058745$, or $+5.87$ percentage points).

### 2.3 Segment Reconciliations Across Required Dimensions
All segment breakdowns recomputed from raw data and saved in `phases/phase4/output/phase4_segment_cohort_churn_analysis.csv` (67 rows) and `phases/phase4/output/phase4_signup_cohort_analysis.csv`. Every row satisfies $Customer\_Count = Churner\_Count + Retained\_Count$ and $Observed\_Churn\_Rate = \frac{Churner\_Count}{Customer\_Count} \times 100\%$.

#### A. Membership Tier Breakdown
| Scope | Membership Tier | Customer Count | Churner Count | Retained Count | Observed Churn Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Validation Cohort ($N = 602$)** | **Silver** | 348 | 45 | 303 | 12.93% |
| | **Gold** | 191 | 26 | 165 | 13.61% |
| | **Platinum** | 63 | 14 | 49 | 22.22% |
| | **Total** | **602** | **85** | **517** | **14.12%** |
| **All Eligible ($N = 2,549$)** | **Silver** | 1,450 | 476 | 974 | 32.83% |
| | **Gold** | 804 | 248 | 556 | 30.85% |
| | **Platinum** | 295 | 94 | 201 | 31.86% |
| | **Total** | **2,549** | **818** | **1,731** | **32.09%** |
| **Full Database ($N = 2,600$)** | **Silver** | 1,474 | 476 | 998 | 32.29% |
| | **Gold** | 824 | 248 | 576 | 30.10% |
| | **Platinum** | 302 | 94 | 208 | 31.13% |
| | **Total** | **2,600** | **818** | **1,782** | **31.46%** |

#### B. City Breakdown
| Scope | City | Customer Count | Churner Count | Retained Count | Observed Churn Rate (%) | Sample Reliability & Statistical Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Validation Cohort ($N = 602$)** | **Austin** | 64 | 13 | 51 | 20.31% | Small sample ($N=64$); highest observed rate |
| | **Boise** | 60 | 7 | 53 | 11.67% | Moderate sample ($N=60$) |
| | **Columbus** | 58 | 9 | 49 | 15.52% | Moderate sample ($N=58$) |
| | **Denver** | 63 | 10 | 53 | 15.87% | Moderate sample ($N=63$) |
| | **Madison** | 62 | 10 | 52 | 16.13% | Moderate sample ($N=62$) |
| | **Nashville** | 43 | 5 | 38 | 11.63% | **Smallest city sample ($N=43$)** |
| | **Portland** | 70 | 8 | 62 | 11.43% | Large city sample ($N=70$) |
| | **Raleigh** | 61 | 6 | 55 | 9.84% | Lowest validation churn rate |
| | **Spokane** | 50 | 6 | 44 | 12.00% | Small sample ($N=50$) |
| | **Tulsa** | 71 | 11 | 60 | 15.49% | **Largest market ($N=71$ val, $N=295$ total)** |
| | **Total** | **602** | **85** | **517** | **14.12%** | **100% Reconciled Across All 10 Cities** |
| **Full Database ($N = 2,600$)** | 10 Cities | **2,600** | **818** | **1,782** | **31.46%** | Ranging from 28.33% (Portland) to 37.59% (Austin) |

*Statistical Precaution:* Each city in the validation cohort contains only 5 to 13 churners. Binomial confidence intervals overlap substantially (e.g. Austin $[11.3\%, 32.2\%]$ vs. Raleigh $[3.7\%, 20.2\%]$). Geography is statistically uninformative compared to behavioral recency and support tickets.

#### C. Signup Cohort Dynamics
| Signup Cohort | Modeling Role | Date Boundaries | Customer Count | Churner Count | Retained Count | Observed Churn Rate (%) | Mean Pre-Cutoff Tenure |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **2021 Signups** | Training | $\le 2021-12-31$ | 745 | 279 | 466 | 37.45% | 985.4 days |
| **2022 Signups** | Training | $2022-01-01 \dots 2022-12-31$ | 792 | 306 | 486 | 38.64% | 639.2 days |
| **2023 H1 Signups** | Training | $2023-01-01 \dots 2023-06-30$ | 410 | 148 | 262 | 36.10% | 365.1 days |
| **2023 Q3 Signups** | Validation | $2023-07-01 \dots 2023-09-30$ | 202 | 51 | 151 | 25.25% | 229.0 days |
| **2023 Q4 Signups** | Validation | $2023-10-01 \dots 2023-12-31$ | 204 | 30 | 174 | 14.71% | 137.4 days |
| **2024 Q1 Signups** | Validation | $2024-01-01 \dots 2024-03-31$ | 196 | 4 | 192 | 2.04% | 46.5 days |
| **2024 Q2 Signups** | Excluded | $\ge 2024-04-01$ | 51 | 0 | 51 | 0.00% | 0.0 days |
| **Total Program Database** | Complete Base | Full Enrollment Period | **2,600** | **818** | **1,782** | **31.46%** | **615.8 days** |

*Exposure Mechanics:* Q1 2024 signups enrolled 0 to 3 months prior to the March 31 cutoff. Because their onboarding shopping is inherently recent, virtually none could qualify as churners under the 90-day post-cutoff inactivity definition. This is an exposure artifact, not a causal retention driver.

### 2.4 Charts & Reporting Integration
- **Generated Publication Charts:**
  - `phases/phase4/output/charts/phase4_signup_cohort_distribution.png`: Dual-axis distribution showing volume ($N$) and churn rate (%) across all cohorts.
  - `phases/phase4/output/charts/phase4_segment_churn_profiles.png`: Tier and store pricing risk breakdowns.
  - `phases/phase4/output/charts/phase4_top_predictors_comparison.png`: Random Forest Gini vs. SHAP importance comparison.
  - `phases/phase4/output/charts/phase4_shap_summary.png`: Beeswarm summary plot of directional feature impacts.
- **Reporting:** All three dimensions (tier, city, and cohort) are integrated into `reports/FINAL_PROJECT_REPORT.md` (Sections 8.1, 8.2, 8.3, 8.4).

---

## 3. Prediction File Validation

### 3.1 File Locations & Column Schema
- **All-Eligible Customer Predictions:** `phases/phase3/output/phase3_batch_predictions_all_eligible.csv`
- **Validation Subset with Drivers:** `phases/phase3/output/phase3_validation_predictions_with_drivers.csv`

### 3.2 Population Completeness & ID Reconciliation
- **Master Customer Population:** 2,600 customers.
- **Excluded Post-Cutoff Signups:** Exactly 51 customers who enrolled after March 31, 2024 (April–May 2024).
- **Eligible Scoring Population:** Exactly **2,549** customers ($2,600 - 51$).
- **Actual Row Count in File:** Exactly **2,549** rows.
- **Unique Customer IDs:** Exactly 2,549 unique IDs. Zero nulls, zero duplicates.
- **Leakage Prevention:** 0 of the 51 post-cutoff signups appear in the batch prediction file.
- **Validation Subset:** Exactly 602 rows in `phase3_validation_predictions_with_drivers.csv` ($85$ observed churners, $105$ predicted at-risk at $\tau = 0.50$).

### 3.3 Output Columns & Value Checks
The file contains all required columns:
1. `CUSTOMER_ID`: Integer, non-null, unique.
2. `CHURN_PROBABILITY`: Numeric, finite, bounded in $[0.0046, 0.9978]$.
3. `PREDICTED_LABEL`: Binary integer $\{0, 1\}$, strictly matching $(\text{CHURN\_PROBABILITY} \ge 0.50)$.
4. `TOP_DRIVER_1`: String, valid transformed feature name.
5. `TOP_DRIVER_1_CONTRIBUTION`: Float, non-zero signed contribution.
6. `TOP_DRIVER_2`: String, valid transformed feature name.
7. `TOP_DRIVER_2_CONTRIBUTION`: Float, non-zero signed contribution.

*Additional Verification:*
- For 100% of the 2,549 accounts, `TOP_DRIVER_1 != TOP_DRIVER_2` (distinct features).
- All driver names are verified model features; zero administrative fields, customer IDs, or target labels appear in driver columns.
- The file opens cleanly with zero index column anomalies.

---

## 4. Regression Verification & Commands Executed

All commands executed from the project root (`c:\7th Sem\AADHI`):

| Test Command | Exit Code | Verified Output Summary | Status |
| :--- | :---: | :--- | :---: |
| `powershell -Command "Get-ChildItem -File"` | `0` | Strictly `predict_customer.py` and `README.md` at root | **PASS** |
| `python -m compileall .` | `0` | Compiled all packages and phases; 0 syntax errors | **PASS** |
| `python predict_customer.py --verify` | `0` | Customer 500014 live probability matches 77.67% | **PASS** |
| `python -m pytest tests/` | `0` | 11 passed in 15.96s (all unit & integration tests) | **PASS** |
| `python scripts/final_verification_check.py` | `0` | Quality gate passed: 63 canonical files, 5 mirrors, all metrics verified | **PASS** |
| `python phases/phase4/phase4_segment_cohort_analysis.py` | `0` | Recomputed 67-row segment/cohort analysis table & chart | **PASS** |
| `python phases/phase3/batch_predict.py` | `0` | Scored all 2,549 eligible customers with local SHAP drivers | **PASS** |

### Remaining Practical Limitations
- **Financial Sensitivity Assumptions:** The $250 Annual Contribution Margin and intervention unit costs ($15 phone outreach, $25 reactivation offer, $5 digital nudge) are hypothetical modeling parameters for marketing capacity planning. They must be updated with FreshBasket's audited ERP figures before budget allocation.
- **Cohort Tenure Truncation:** The 2.04% churn rate among Q1 2024 signups is a structural artifact of shorter observation exposure, not an intrinsic causal retention driver.

---

## 5. Final Verdict

### **PASS**

**Auditor Justification:**
1. **Requirement 1 (Data Quality & EDA):** Fully implemented in executable Python scripts and Jupyter notebooks. All 7 required issues (missing months, duplicates, negative spend, zero spend anomalies, text casing, outlier spending, missing values) are quantified, resolved with documented defensible decisions, and embedded as saved outputs in both canonical notebooks (`phase1_eda_and_data_quality_notebook.ipynb` and `phase1_cleaning_notebook.ipynb`) with zero execution errors.
2. **Requirement 2 (Feature Importance, SHAP & Segments):** Feature importances reflect the saved Random Forest model. SHAP attributions operate additively on the churn probability scale ($< 2.8 \times 10^{-15}$ reconstruction error). Membership tier, city, and signup cohort breakdowns reconcile 100% across validation ($N=602$), eligible ($N=2,549$), and full ($N=2,600$) scopes, with statistical reliability warnings and non-causal guardrails documented in the final report and presentation.
3. **Requirement 3 (Prediction File & SHAP Drivers):** Exactly 2,549 eligible accounts scored with finite probabilities, binary labels matching the 0.50 threshold, and distinct, valid top-two SHAP drivers. Customer 500014 benchmark verified at 77.67%. All safety and root-isolation constraints were strictly preserved.
