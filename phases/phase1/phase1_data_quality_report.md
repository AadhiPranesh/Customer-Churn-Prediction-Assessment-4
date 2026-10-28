# Phase 1: Data Quality and Remediation Report

**Project:** Customer Churn Prediction for FreshBasket Retail Loyalty Program  
**Assessment:** Propel Customer Churn Assessment  

**Date:** October 2026  
**Scope:** Phase 1 (Data Inspection, Cleaning, Remediation, and Target Verification)

---

## 1. Dataset Overview and Relational Architecture

The FreshBasket Retail loyalty dataset was provided as an Excel workbook (`FreshBasket_Loyalty_Churn_Dataset.xlsx`). Programmatic inspection confirmed that row 1 across all sheets contains a decorative banner title (`FreshBasket Loyalty: Customer Churn Prediction`). Real table headers begin at **row 2** (`header=1` in pandas).

### 1.1 Ingested Table Dimensions and Schema

| Table Name | Ingestion Method | Raw Rows | Raw Columns | Primary Key (PK) / Foreign Key (FK) | Granularity | Date Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `fb Customer Profile` | `header=1` | 2,608 | 10 | PK: `CUSTOMER_ID` | 1 row per loyalty member | Signup dates: `2021-02-18` to `2024-05-02` |
| `fb Churn Label` | `header=1` | 2,605 | 6 | PK: `CUSTOMER_ID` | 1 row per loyalty member | Observation end: `2024-06-30` (constant) |
| `fb Monthly Activity` | `header=1` | 28,073 | 12 | Composite PK: (`CUSTOMER_ID`, `MONTH`) | 1 row per customer per month | Calendar months: `2023-01-01` to `2024-06-01` (18 mos) |

### 1.2 Relational Integrity and Entity Overlap
- **Customer Overlap:** `set(Customer Profile) == set(Churn Label)`. Exactly **2,600 unique customer IDs** exist in both tables. There are zero orphaned keys between Profile and Churn.
- **Activity Table Coverage:** Exactly **2,594 unique customer IDs** appear in `fb Monthly Activity`. Exactly **6 customers** in the Profile/Churn tables have **zero records** in `fb Monthly Activity` (`500754`, `501092`, `501216`, `501563`, `502014`, `502550`). All 6 are flagged with `INSUFFICIENT_HISTORY_FLAG = 1` in `fb Churn Label`.

---

## 2. Detailed Data Quality Issues, Impact, and Treatments

### Issue 1: Exact Duplicate Records Across Tables
1. **Records Affected:**
   - `fb Customer Profile`: 8 duplicate rows (16 rows involved).
   - `fb Churn Label`: 5 duplicate rows (10 rows involved).
   - `fb Monthly Activity`: 58 exact full-row duplicates (116 rows involved).
2. **Likely Impact:** If unaddressed, joins between tables cause Cartesian row multiplication, distorting customer counts, inflate spend aggregates, and violate statistical independence.
3. **Treatment Chosen:** Applied exact deduplication (`drop_duplicates()`) across all three tables.
4. **Reason for Treatment:** Every field in these duplicate pairs was 100% identical. Removing redundant entries restores unique primary keys (`CUSTOMER_ID` in Profile/Churn; `CUSTOMER_ID` + `MONTH` in Activity) without losing any distinct information.

---

### Issue 2: Conflicting Customer-Month Duplicates in Monthly Activity
1. **Records Affected:** Exactly 2 pairs of rows (4 rows total) in `fb Monthly Activity`:
   - Customer `501780`, Month `2024-05-01`: Row 14414 had `TOTAL_SPEND = 116.18`, Row 16384 had `TOTAL_SPEND = -116.18`. All other fields (`TRANSACTIONS=3`, `AVG_BASKET_VALUE=38.73`, `APP_SESSIONS=6`, etc.) were identical.
   - Customer `502065`, Month `2023-04-01`: Row 26567 had `TOTAL_SPEND = 70.32`, Row 15432 had `TOTAL_SPEND = -70.32`. All other fields (`TRANSACTIONS=3`, `AVG_BASKET_VALUE=23.44`, `APP_SESSIONS=0`, etc.) were identical.
2. **Likely Impact:** Creates multiple records for the same customer-month with contradictory spend directions, violating composite primary key uniqueness.
3. **Treatment Chosen:** Retained the valid positive spend record and discarded the negative duplicate entry.
4. **Reason for Treatment:** In both cases, the positive spend entry perfectly matched the basket value ($3 \times 38.73 = 116.19$ and $3 \times 23.44 = 70.32$). The negative entry was a duplicate data-entry error.

---

### Issue 3: Negative Spend Records in Monthly Activity
1. **Records Affected:** 24 rows initially. After resolving the 2 conflicting duplicate pairs above, exactly **22 standalone negative spend rows** remain in `fb Monthly Activity`.
2. **Likely Impact:** Produces negative cumulative spend in RFM features, distorts trend slopes, and misrepresents customer monetary value.
3. **Treatment Chosen:**
   - Did not blindly overwrite the original column.
   - Preserved `TOTAL_SPEND` in its raw form.
   - Added an audit flag `IS_NEGATIVE_SPEND = 1`.
   - Created a rectified column `TOTAL_SPEND_RECONSTRUCTED = abs(TOTAL_SPEND)`.
4. **Reason for Treatment:** Exhaustive inspection revealed that in **100% of these 22 cases**:
   - `TRANSACTIONS` is positive ($1 \dots 10$).
   - `AVG_BASKET_VALUE` is strictly positive ($16.75 \dots 52.38$).
   - `abs(TOTAL_SPEND) / TRANSACTIONS == AVG_BASKET_VALUE` within rounding precision ($< 0.005$).
   - The official Data Dictionary explicitly states: *"Total spend in the month, USD (a few negative/zero data-entry errors present)"*.
   This confirms that the negative sign was a clerical sign inversion. Providing both raw and reconstructed columns ensures full auditability and transparency.

---

### Issue 4: Records with Transactions > 0 but Zero Spend
1. **Records Affected:** Exactly 20 rows in `fb Monthly Activity`.
2. **Likely Impact:** Understates total member revenue; creates a contradiction where shopping visits occurred with zero dollars spent, despite non-zero average basket values.
3. **Treatment Chosen:**
   - Preserved `TOTAL_SPEND` as raw.
   - Added an audit flag `IS_ZERO_SPEND_WITH_TXNS = 1`.
   - In `TOTAL_SPEND_RECONSTRUCTED`, imputed spend as `round(TRANSACTIONS * AVG_BASKET_VALUE, 2)`.
4. **Reason for Treatment:** In all 20 rows, `TRANSACTIONS` is positive ($1 \dots 11$) and `AVG_BASKET_VALUE` is non-zero (\$15.12 to \$48.69). The member shopped and items were priced, but total spend failed to record. Reconstructing total spend via transaction count $\times$ average basket value aligns the accounting without fabricating arbitrary numbers.

---

### Issue 5: Inconsistent Casing and Whitespace in Categoricals
1. **Records Affected:**
   - `CITY`: 40 raw string variations for only 10 actual metropolitan areas (e.g., `' Spokane '`, `'SPOKANE'`, `'spokane'`).
   - `MEMBERSHIP_TIER`: 12 raw string variations for 3 actual loyalty tiers (`Silver`, `Gold`, `Platinum`).
2. **Likely Impact:** Categorical aggregations and one-hot encoding treat `'Spokane'` and `'SPOKANE'` as distinct categories, artificially inflating feature dimensionality and fragmenting sample sizes.
3. **Treatment Chosen:** Applied `.astype(str).str.strip().str.title()` to standardize strings into clean Title Case, while retaining raw columns (`CITY_RAW`, `MEMBERSHIP_TIER_RAW`) for verification.
4. **Reason for Treatment:** Clean normalization collapses the data into exactly 10 distinct cities and 3 distinct tiers, directly matching the business reality described in the assessment.

---

### Issue 6: Missing Demographics in Customer Profile
1. **Records Affected:**
   - `AGE`: 78 missing values ($2.99\%$).
   - `GENDER`: 52 missing values ($1.99\%$).
   - `HOME_STORE_PRICE_TIER`: 130 missing values ($4.98\%$).
2. **Likely Impact:** Standard statistical models will drop rows with missing values if unhandled, leading to selection bias and the loss of up to 260 members ($10\%$).
3. **Treatment Chosen:**
   - Preserved all 2,600 customer rows (no row deletions).
   - In `fb Customer Profile`: Created `GENDER_CLEANED` with an explicit `'Unknown'` category.
   - Created `HOME_STORE_PRICE_TIER_CLEANED` with an explicit `'Unknown'` category.
   - Added a binary indicator `AGE_IS_MISSING = 1` while preserving `AGE` as float for EDA.
4. **Reason for Treatment:** Missingness in loyalty demographics is rarely missing completely at random (MCAR). Encoding explicit `'Unknown'` categories allows models to evaluate whether non-reporting of demographic data is itself correlated with customer disengagement.

---

### Issue 7: Customers with No Monthly Activity Records
1. **Records Affected:** Exactly 6 customer IDs (`500754`, `501092`, `501216`, `501563`, `502014`, `502550`).
2. **Likely Impact:** Performing an inner join between Profile and Activity would silently drop these 6 customers, violating the official churn outcome evaluation set ($N=2,600$).
3. **Treatment Chosen:** Kept these customers in the master population. Left-join will be used in Phase 2, imputing zero for transaction and spend counts, accompanied by an `IS_ZERO_ACTIVITY_CUSTOMER = 1` flag.
4. **Reason for Treatment:** All 6 members have ground-truth churn labels (2 churned, 4 retained) and all 6 have `INSUFFICIENT_HISTORY_FLAG = 1`. Dropping them would invalidate evaluation metrics.

---

## 3. Churn Target Verification

The official assessment definition states:
> *"The churn target (CHURNED, in fb Churn Label) is defined as zero purchase transactions in the final three months of the window (April-June 2024) despite the member having purchase history earlier in the window."*

### 3.1 Verification Cross-Tabulation
We calculated total purchase transactions for every customer during April, May, and June 2024 from `fb Monthly Activity` and cross-tabulated against the ground-truth `CHURNED` column:

| Target Window Transactions (Apr–Jun 2024) | `CHURNED = 0` (Retained) | `CHURNED = 1` (Churned) | Total Members |
| :--- | :--- | :--- | :--- |
| **Transactions $> 0$ (Active Shoppers)** | **1,657** | **0** | **1,657** |
| **Transactions $== 0$ (Zero Shoppers)** | **125** | **818** | **943** |
| **Total** | **1,782** | **818** | **2,600** |

*(Note: Plus the 6 members missing from activity: 4 with `CHURNED=0`, 2 with `CHURNED=1`, bringing total to 1,782 retained and 818 churned).*

### 3.2 Explanation of the 125 Non-Churners with Zero Target Transactions
Why are 125 customers with zero purchases in April–June 2024 labelled `CHURNED = 0`?
1. **No Prior Purchase History (120 members):** Exactly 120 of these customers have `LAST_PURCHASE_DATE = NaT` and zero transactions across all 15 pre-cutoff months. Because they never established a shopping history, they cannot "churn" from a habit they never formed. They are inactive accounts or non-activated signups.
2. **Edge-Case Activity Discrepancies (5 members):** Exactly 5 members (`500994`, `501186`, `501491`, `502008`, `502024`) have `CHURNED = 0` and had `LAST_PURCHASE_DATE` recorded in April, May, or June 2024 in `fb Churn Label`, yet their activity records show 0 transactions in those months. As instructed, **we did not silently alter these labels**, but flagged them for formal documentation.

---

## 4. Prediction Cutoff and Target Leakage Guardrails

The intended prediction cutoff is **March 31, 2024 (23:59:59)**. The outcome window is **April 1, 2024 through June 30, 2024**.

### 4.1 Identified Post-Cutoff Fields (Leakage Risks for Phase 2)
The following fields in the raw dataset contain post-cutoff information and must **never** be used as predictive features:
1. `MONTH >= 2024-04-01` in `fb Monthly Activity`: Any spend, transaction, ticket, or app activity occurring in April, May, or June 2024.
2. `LAST_PURCHASE_DATE` in `fb Churn Label`: Recorded as of June 30, 2024. If `LAST_PURCHASE_DATE >= 2024-04-01`, churn is guaranteed to be 0.
3. `MONTHS_OBSERVED` in `fb Churn Label`: Computed across the full 18-month window through June 2024.
4. `OBSERVATION_END_DATE` in `fb Churn Label`: Constant metadata (`2024-06-30`).
5. `INSUFFICIENT_HISTORY_FLAG` in `fb Churn Label`: Derived as of June 30, 2024.

In Phase 2, all behavioral features must be computed **strictly from activity records where `MONTH < 2024-04-01`**.

---

## 5. Cleaning Audit Trail and Record Count Validation

| Table | Initial Raw Rows | Cleaning Step | Records Dropped / Modified | Final Cleaned Rows | Unique Customer IDs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fb Customer Profile` | 2,608 | `drop_duplicates()` | 8 duplicate rows dropped | **2,600** | **2,600** |
| `fb Churn Label` | 2,605 | `drop_duplicates()` | 5 duplicate rows dropped | **2,600** | **2,600** |
| `fb Monthly Activity` | 28,073 | `drop_duplicates()` | 58 exact duplicate rows dropped | 28,015 | 2,594 |
| `fb Monthly Activity` | 28,015 | Resolve 2 +/- spend duplicate pairs | 2 negative duplicate rows dropped | **28,013** | **2,594** |

### Output Files Saved:
1. `phase_1_outputs/cleaned_data/cleaned_fb_customer_profile.csv` (2,600 rows, 14 columns)
2. `phase_1_outputs/cleaned_data/cleaned_fb_churn_label.csv` (2,600 rows, 6 columns)
3. `phase_1_outputs/cleaned_data/cleaned_fb_monthly_activity.csv` (28,013 rows, 15 columns)

Zero customers were deleted. Zero target labels were modified. Full reproducibility is guaranteed.
