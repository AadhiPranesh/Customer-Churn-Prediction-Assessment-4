# Phase 2: Data Checks and Integrity Audit

**Project:** Customer Churn Prediction for FreshBasket Retail Loyalty Program  
**Assessment:** Propel Internship Assessment  

**Artifact Audited:** `phase2_modeling_dataset.csv` (2,600 rows $\times$ 33 columns)  
**Prediction Cutoff:** March 31, 2024 (23:59:59)  
**Scope:** Phase 2 Data Quality, Row Uniqueness, Cutoff Compliance, and Target Integrity

---

## 1. Row Uniqueness and Entity Integrity Checks

| Verification Check | Target / Expected Value | Observed Result | Status | Evidence / Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Total Row Count** | 2,600 | **2,600** | **PASSED** | Exactly matches the unique loyalty member population. |
| **Unique Customer IDs** | 2,600 | **2,600** | **PASSED** | `df['CUSTOMER_ID'].nunique() == 2600`. |
| **Duplicate Customer IDs** | 0 | **0** | **PASSED** | Zero duplicate keys detected. Granularity is strictly 1 row per customer. |
| **Primary Key Overlap** | 100% | **100%** | **PASSED** | Matches `cleaned_fb_customer_profile.csv` and `cleaned_fb_churn_label.csv` exactly. |

---

## 2. Target Label Consistency Checks

| Target Class | Cleaned Official Count | `phase2_modeling_dataset.csv` Count | Percentage (%) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Retained (`CHURNED = 0`)** | 1,782 | **1,782** | **68.54%** | **PASSED** |
| **Churned (`CHURNED = 1`)** | 818 | **818** | **31.46%** | **PASSED** |
| **Total Population** | **2,600** | **2,600** | **100.0%** | **PASSED** |

* **Zero Silent Modifications:** Ground-truth labels from `fb Churn Label` match with 100% fidelity.
* **Target Isolation:** `CHURNED` is cleanly isolated as the target vector $y$. It is **not** used in the calculation of any predictive feature.

---

## 3. Prediction Cutoff Compliance & Target Leakage Audit

* **Cutoff Date Enforced:** **March 31, 2024 (23:59:59)**.
* **Pre-Cutoff Monthly Records Processed:** Exactly **23,253 monthly activity records** (spanning `2023-01-01` to `2024-03-01`, 15 months).
* **Target Window Exclusion:** All **4,760 monthly activity records** dated April, May, or June 2024 were **strictly filtered out** prior to feature aggregation.
* **Post-Cutoff / Leaked Field Exclusion Audit:**
  - `LAST_PURCHASE_DATE`: **EXCLUDED** (Recorded as of June 30, 2024; replaced by `RECENCY_MONTHS_PRE_CUTOFF`).
  - `MONTHS_OBSERVED`: **EXCLUDED** (Calculated through June 2024; replaced by `PRE_CUTOFF_OBSERVED_MONTHS` and `ELIGIBLE_PRE_CUTOFF_MONTHS`).
  - `INSUFFICIENT_HISTORY_FLAG`: **EXCLUDED** (End-of-window derivation; replaced by `IS_LIMITED_PRE_CUTOFF_HISTORY`).
  - `OBSERVATION_END_DATE`: **EXCLUDED** (Constant June 30, 2024 metadata).
  - `CUSTOMER_ID`: **RETAINED AS METADATA ONLY** (Excluded from model training feature matrices).

---

## 4. Missing Values Audit Across All 33 Columns

```
========================================================================================
                          MISSING VALUE AUDIT SUMMARY
========================================================================================
Column Name                            Data Type       Missing Count    Missing Pct (%)
----------------------------------------------------------------------------------------
AGE                                    Float                 78             2.99%
All other 32 columns                   Various                0             0.00%
========================================================================================
```

* **Handling of Missing Age:** Exactly 78 members have unrecorded age ($2.99\%$). This missingness is preserved as `NaN` in `AGE` while being explicitly flagged by `AGE_IS_MISSING = 1`. In Phase 3, standard median imputation or missing-indicator pipelines will handle this cleanly without row loss.
* **Categorical Imputation:** Categorical columns (`GENDER`, `HOME_STORE_PRICE_TIER`) have zero `NaN` values, with unrecorded categories cleanly represented as `'Unknown'`.

---

## 5. Numerical Distribution and Outlier Inspection

Summary statistics across key engineered features in `phase2_modeling_dataset.csv`:

| Feature Name | Count | Mean | Median | Min | Max | Plausibility & Sanity Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `TENURE_DAYS_PRE_CUTOFF` | 2,600 | 562.9 | 562.0 | 0 | 1,137 | Spans 0 days (enrolled on cutoff) to ~3.1 years. Clipped at 0 for post-cutoff signups. |
| `ELIGIBLE_PRE_CUTOFF_MONTHS`| 2,600 | 13.5 | 15.0 | 0 | 15 | Calendar tenure months in observation window. Denominator for monthly rates. |
| `TOTAL_TXNS_PRE_CUTOFF` | 2,600 | 36.9 | 30.0 | 0 | 165 | Plausible multi-month grocery trip counts. Max ~11 trips/month. |
| `TOTAL_SPEND_PRE_CUTOFF` | 2,600 | \$1,193.89 | \$913.57 | \$0.00 | \$7,841.17 | Cumulative spend over 15 months. Zero negative values. |
| `AVG_MONTHLY_TXNS_PRE_CUTOFF`| 2,600 | 2.90 | 2.67 | 0.0 | 13.0 | Monthly transaction rate divided by eligible calendar months. |
| `AVG_MONTHLY_SPEND_PRE_CUTOFF`| 2,600 | \$93.94 | \$83.12 | \$0.00 | \$522.74 | Monthly spending rate divided by eligible calendar months. Accounts for dormancy. |
| `AVG_BASKET_VALUE_PRE_CUTOFF` | 2,600 | \$29.50 | \$30.71 | \$0.00 | \$81.68 | Typical grocery basket averages. |
| `TXNS_LAST_3M` | 2,600 | 7.9 | 7.0 | 0 | 37 | Captures recent Q1 2024 transaction momentum. |
| `SPEND_LAST_3M` | 2,600 | \$257.59 | \$193.68 | \$0.00 | \$4,060.06 | Captures recent Q1 2024 monetary momentum. |
| `RECENCY_MONTHS_PRE_CUTOFF` | 2,600 | 11.0 | 0.0 | 0 | 99 | Median is 0 (purchased in March 2024). 99 represents sentinel for never purchased. |
| `TOTAL_SUPPORT_TICKETS_PRE_CUTOFF` | 2,600 | 1.40 | 1.0 | 0 | 8 | Pre-cutoff customer service interactions. |
| `AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF`| 2,600 | 1.45 | 1.33 | 0.0 | 11.0 | Mean monthly digital app engagement over eligible months. |

---

## 6. Population Accounting: Included vs. Excluded Records

```
========================================================================================
                          CUSTOMER POPULATION ACCOUNTING
========================================================================================
Customer Sub-Group                    Count    Pct (%)   Target Split (0 / 1)   Evaluation Action
----------------------------------------------------------------------------------------
Eligible Mature Signups (<= Jun 2023) 1,947     74.88%   1,214 Ret / 733 Churn  Assigned to TRAIN cohort
Eligible Newer Signups (Jul 23-Mar 24)  602     23.15%     517 Ret /  85 Churn  Assigned to VAL cohort
Post-Cutoff Signups (> Mar 31, 2024)     51      1.96%      51 Ret /   0 Churn  EXCLUDED from scoring
----------------------------------------------------------------------------------------
Total Dataset Population              2,600    100.00%   1,782 Ret / 818 Churn  Retained in CSV
========================================================================================
```

### Rationale for Post-Cutoff Exclusion:
* Exactly **51 members** enrolled after March 31, 2024.
* Because they were not yet members at the cutoff, they could not have been scored in a real-world predictive deployment.
* Retained in `phase2_modeling_dataset.csv` for 100% population reconciliation, but assigned `VALIDATION_COHORT = 'EXCLUDED_POST_CUTOFF'` and flagged with `IS_POST_CUTOFF_SIGNUP = 1`.
* **Zero customers were deleted from the dataset.**
