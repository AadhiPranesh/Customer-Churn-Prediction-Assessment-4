# Phase 1: Key Methodological Decisions and Pre-Phase 2 Sign-Off

**Project:** Customer Churn Prediction for FreshBasket Retail Loyalty Program  
**Assessment:** Propel Customer Churn Assessment  

**Date:** October 2026  

---

This document summarizes the specific data-cleaning and methodological decisions made during **Phase 1** that require explicit awareness and sign-off before commencing **Phase 2 (Customer-Level Feature Engineering)**.

---

### Decision 1: Treatment of Negative and Zero-Spend Records
* **Context:** In `fb Monthly Activity`, 22 standalone records contain negative spend values (e.g. $-\$226.93$), and 20 records contain zero spend with non-zero transactions. In all 42 records, `AVG_BASKET_VALUE` is strictly positive and matches the true transaction economics ($|\text{TOTAL\_SPEND}| / \text{TRANSACTIONS} \approx \text{AVG\_BASKET\_VALUE}$).
* **Action Taken in Phase 1:** We preserved `TOTAL_SPEND` in its raw form, added explicit audit flags (`IS_NEGATIVE_SPEND = 1`, `IS_ZERO_SPEND_WITH_TXNS = 1`), and created `TOTAL_SPEND_RECONSTRUCTED` which corrects the sign and imputes spend from basket values.
* **Proposed Phase 2 Decision:** Use `TOTAL_SPEND_RECONSTRUCTED` as the basis for all monetary RFM features and spend velocity calculations, ensuring no customer is assigned a negative lifetime spend.
* **Decision Needed:** Confirm use of `TOTAL_SPEND_RECONSTRUCTED` for downstream feature aggregations.

---

### Decision 2: Handling of 373 Members with Insufficient History
* **Context:** In `fb Churn Label`, 373 loyalty members have `INSUFFICIENT_HISTORY_FLAG = 1` ($< 4$ months of observed history). This includes:
  - Exactly **6 members** who have zero monthly activity records.
  - Exactly **120 members** who never made a single purchase during the observation window (`LAST_PURCHASE_DATE = NaT`), all of whom are labelled `CHURNED = 0`.
* **Action Taken in Phase 1:** All 2,600 members were preserved in the cleaned customer profile and churn tables. No rows were deleted.
* **Proposed Phase 2 Decision:** Retain all 2,600 members in the modeling dataset to respect the client's official evaluation population. For the 6 members with no activity, impute 0 for transaction/spend counts, and engineer an explicit feature `IS_INSUFFICIENT_HISTORY = 1` to allow models to learn their specific baseline hazard.
* **Decision Needed:** Confirm whether these 373 members should be included in the primary model or segmented into a distinct "New Account Activation" sub-model.

---

### Decision 3: The 5 Discrepant Ground-Truth Churn Labels
* **Context:** Exactly 5 customers (`500994`, `501186`, `501491`, `502008`, `502024`) are labelled `CHURNED = 0` in `fb Churn Label` and have `LAST_PURCHASE_DATE` recorded in April, May, or June 2024, yet their records in `fb Monthly Activity` show zero purchase transactions during those target months.
* **Action Taken in Phase 1:** Preserved the original ground-truth label `CHURNED = 0` without alteration, in strict accordance with the assessment rule: *"Do not silently change target labels. If any label cannot be verified, document it for review."*
* **Proposed Phase 2 Decision:** Maintain these official client labels without modification to preserve scoring compatibility with client test keys.
* **Decision Needed:** Formal sign-off on keeping official labels unchanged despite minor activity table discrepancy.

---

### Decision 4: Cohort Validation Splitting Boundary
* **Context:** The assessment explicitly requires a time-based or cohort-based validation split. Historical backtesting churn labels do not exist in the workbook.
* **Proposed Phase 2 Decision:** Partition the 2,600 members into:
  - **Training Cohort (~75%):** Loyalty members who enrolled on or before **June 30, 2023** ($N \approx 1,939$).
  - **Validation Cohort (~25%):** Loyalty members who enrolled between **July 1, 2023 and March 31, 2024** ($N \approx 661$).
  Both cohorts will be evaluated strictly on their behavior prior to March 31, 2024, predicting the April–June 2024 outcome window.
* **Decision Needed:** Approval of the **June 30, 2023** enrollment cutoff boundary.

---

### Decision 5: Standardization of Categorical Values
* **Context:** `CITY` contained 40 raw text variations and `MEMBERSHIP_TIER` contained 12 variations due to whitespace and mixed capitalization.
* **Action Taken in Phase 1:** Standardized into 10 clean city names and 3 clean tier names (`Silver`, `Gold`, `Platinum`).
* **Proposed Phase 2 Decision:** Use these 10 clean cities and 3 clean tiers in one-hot encoding pipelines, with `Unknown` categories for missing values.
* **Decision Needed:** Confirmation of categorical schema standardization.
