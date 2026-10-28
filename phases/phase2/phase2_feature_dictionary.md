# Phase 2: Feature Dictionary — FreshBasket Customer Churn Project

**Project:** Customer Churn Prediction for FreshBasket Retail Loyalty Program  
**Assessment:** Propel Internship Assessment  
 
**Dataset Generated:** `phase2_modeling_dataset.csv` (2,600 rows $\times$ 33 columns)  
**Prediction Cutoff:** March 31, 2024 (23:59:59)  
**Strict Leakage Policy:** All behavioral features derived strictly from records where `MONTH < 2024-04-01`.

---

## 1. Overview of Feature Blocks

The modeling dataset consolidates customer profile information and 15 months of pre-cutoff monthly behavioral history into **33 structured columns**:

| Feature Block | Number of Columns | Primary Source | Business Function |
| :--- | :---: | :--- | :--- |
| **Record Identifiers & Metadata** | 2 | `fb Customer Profile` | Entity tracking and temporal audit (excluded from model input). |
| **Customer Profile & Demographics** | 8 | `fb Customer Profile` | Static member attributes, tier, store preferences, tenure, and eligible calendar months. |
| **Purchase Behavior (Pre-Cutoff)** | 9 | `fb Monthly Activity` | Pre-cutoff shopping volume, spend, basket value, momentum, and recency. |
| **Engagement & Support Friction** | 7 | `fb Monthly Activity` | Digital touchpoints (totals & monthly averages) and operational friction (tickets, complaints). |
| **Quality & Limited-History Flags** | 5 | Derived as of 2024-03-31 | Identifies newly enrolled members, missing activity, and inactive signups. |
| **Validation Cohort Assignment** | 1 | Derived from `SIGNUP_DATE` | Primary partition for out-of-cohort model evaluation. |
| **Prediction Target** | 1 | `fb Churn Label` | Binary ground-truth outcome ($1 = \text{Churned}, 0 = \text{Retained}$). |

---

## 2. Comprehensive Column-by-Column Feature Dictionary

| Column Name | Data Type | Feature Block | Source Column(s) | Calculation Logic / Formula | Plain-English Description & Business Rationale |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `CUSTOMER_ID` | Integer | Metadata / ID | `fb Customer Profile.CUSTOMER_ID` | Direct unique identifier. | **Identifier only.** Excluded from model features to prevent arbitrary memorization. |
| `SIGNUP_DATE` | Date (`YYYY-MM-DD`) | Metadata / Temporal | `fb Customer Profile.SIGNUP_DATE` | Date member enrolled in loyalty program. | Used for cohort assignment and tenure verification. Excluded from direct model features. |
| `AGE` | Float | Profile | `fb Customer Profile.AGE` | Reported customer age in years. Preserved with 78 missing values. | Demographic baseline. Younger vs. older shopper loyalty tendencies. |
| `AGE_IS_MISSING` | Integer (0/1) | Quality Flag | Derived from `AGE` | `1 if AGE is null else 0` | Flag indicating whether the customer omitted their age during signup. |
| `GENDER` | String | Profile | `fb Customer Profile.GENDER` | Cleaned categories: `F`, `M`, `Other`, `Unknown`. | Demographic baseline. Captures gender segmentation and non-reporting. |
| `CITY` | String | Profile | `fb Customer Profile.CITY` | Normalized Title Case: 10 clean cities. | Geographic region. Captures local market competition and store operations. |
| `MEMBERSHIP_TIER` | String | Profile | `fb Customer Profile.MEMBERSHIP_TIER` | Normalized: `Silver`, `Gold`, `Platinum`. | Loyalty status. Tests whether premium tier perks insulate against attrition. |
| `HOME_STORE_PRICE_TIER` | String | Profile | `fb Customer Profile.HOME_STORE_PRICE_TIER` | Cleaned: `Value`, `Mainstream`, `Upscale`, `Unknown`. | Price positioning of frequented store. Tests price sensitivity across income segments. |
| `ELIGIBLE_PRE_CUTOFF_MONTHS` | Integer | Profile / History | `SIGNUP_DATE` | Elapsed calendar months from `max(SIGNUP_DATE, 2023-01-01)` to `2024-03-31` (1–15 mos; 0 if post-cutoff). | **Eligible Calendar Months.** Denominator for all monthly rate features. Fairly normalizes for signup dates and accounts for zero-activity dormant months. |
| `TENURE_DAYS_PRE_CUTOFF` | Integer | Profile / Tenure | `SIGNUP_DATE` | `max(0, (2024-03-31 - SIGNUP_DATE).days)` | **Tenure as of Cutoff.** Days elapsed between loyalty enrollment and March 31, 2024. |
| `IS_POST_CUTOFF_SIGNUP` | Integer (0/1) | Quality Flag | `SIGNUP_DATE` | `1 if SIGNUP_DATE > 2024-03-31 else 0` | Flags the 51 customers who enrolled after the prediction cutoff (not available to score). |
| `TOTAL_TXNS_PRE_CUTOFF` | Integer | Purchase Behavior | `fb Monthly Activity.TRANSACTIONS` | $\sum \text{TRANSACTIONS}$ for `MONTH < 2024-04-01` | Cumulative shopping trips prior to cutoff. Overall transactional loyalty. |
| `TOTAL_SPEND_PRE_CUTOFF` | Float | Purchase Behavior | `fb Monthly Activity.TOTAL_SPEND_RECONSTRUCTED` | $\sum \text{TOTAL\_SPEND\_RECONSTRUCTED}$ for `MONTH < 2024-04-01` | Cumulative monetary spend prior to cutoff (using validated reconstructed spend). |
| `ACTIVE_MONTHS_PRE_CUTOFF` | Integer | Purchase Behavior | `fb Monthly Activity.TRANSACTIONS` | Count of pre-cutoff months where $\text{TRANSACTIONS} > 0$. | Regularity of shopping habit. Distinguishes consistent shoppers from one-off buyers. |
| `PRE_CUTOFF_OBSERVED_MONTHS` | Integer | Quality / History | `fb Monthly Activity.MONTH` | Count of pre-cutoff monthly activity rows present. | Number of calendar months with recorded activity in the table. |
| `AVG_MONTHLY_TXNS_PRE_CUTOFF` | Float | Purchase Behavior | `TOTAL_TXNS_PRE_CUTOFF`, `ELIGIBLE_PRE_CUTOFF_MONTHS` | $\frac{\text{TOTAL\_TXNS}}{\text{ELIGIBLE\_MONTHS}}$ if eligible $> 0$ else $0.0$ | Average shopping frequency per eligible month. Reflects true visit rate across tenure. |
| `AVG_MONTHLY_SPEND_PRE_CUTOFF` | Float | Purchase Behavior | `TOTAL_SPEND_PRE_CUTOFF`, `ELIGIBLE_PRE_CUTOFF_MONTHS` | $\frac{\text{TOTAL\_SPEND}}{\text{ELIGIBLE\_MONTHS}}$ if eligible $> 0$ else $0.0$ | Average monetary spend per eligible month. Financial run-rate of the customer across membership. |
| `AVG_BASKET_VALUE_PRE_CUTOFF` | Float | Purchase Behavior | `TOTAL_SPEND_PRE_CUTOFF`, `TOTAL_TXNS_PRE_CUTOFF` | $\frac{\text{TOTAL\_SPEND}}{\text{TOTAL\_TXNS}}$ if txns $> 0$ else $0.0$ | Average dollars spent per transaction visit. Measures grocery basket depth. |
| `TXNS_LAST_3M` | Integer | Purchase Behavior | `fb Monthly Activity.TRANSACTIONS` | $\sum \text{TRANSACTIONS}$ for `2024-01-01 <= MONTH <= 2024-03-01` | Recent visit momentum in Q1 2024. Sharp drops signal imminent churn. |
| `SPEND_LAST_3M` | Float | Purchase Behavior | `fb Monthly Activity.TOTAL_SPEND_RECONSTRUCTED` | $\sum \text{TOTAL\_SPEND}$ for `2024-01-01 <= MONTH <= 2024-03-01` | Recent monetary momentum in Q1 2024. Immediate pre-churn spending erosion. |
| `RECENCY_MONTHS_PRE_CUTOFF` | Integer | Purchase Behavior | `fb Monthly Activity.MONTH`, `TRANSACTIONS` | Months elapsed from 2024-03-01 to latest pre-cutoff month with transactions $> 0$. Sentinel = 99. | **Purchase Recency.** Months since last grocery visit. Higher recency strongly signals defection. |
| `NEVER_PURCHASED_PRE_CUTOFF` | Integer (0/1) | Quality Flag | Derived from `RECENCY_MONTHS_PRE_CUTOFF` | `1 if customer had 0 pre-cutoff purchases else 0` | Distinguishes customers who stopped shopping from inactive signups who never shopped. |
| `TOTAL_APP_SESSIONS_PRE_CUTOFF` | Integer | Engagement | `fb Monthly Activity.APP_SESSIONS` | $\sum \text{APP\_SESSIONS}$ prior to cutoff. | Total mobile app sessions during pre-cutoff window. |
| `TOTAL_EMAILS_OPENED_PRE_CUTOFF` | Integer | Engagement | `fb Monthly Activity.EMAILS_OPENED` | $\sum \text{EMAILS\_OPENED}$ prior to cutoff. | Total marketing emails opened during pre-cutoff window. |
| `AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF` | Float | Engagement | `TOTAL_APP_SESSIONS_PRE_CUTOFF`, `ELIGIBLE_PRE_CUTOFF_MONTHS` | $\frac{\text{TOTAL\_APP}}{\text{ELIGIBLE\_MONTHS}}$ if eligible $> 0$ else $0.0$ | Digital loyalty touchpoint. Mean app sessions per eligible month. |
| `AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF` | Float | Engagement | `TOTAL_EMAILS_OPENED_PRE_CUTOFF`, `ELIGIBLE_PRE_CUTOFF_MONTHS` | $\frac{\text{TOTAL\_EMAIL}}{\text{ELIGIBLE\_MONTHS}}$ if eligible $> 0$ else $0.0$ | Mean marketing emails opened per eligible month. |
| `TOTAL_COUPONS_REDEEMED_PRE_CUTOFF` | Integer | Engagement | `fb Monthly Activity.COUPONS_REDEEMED` | $\sum \text{COUPONS\_REDEEMED}$ prior to cutoff. | Deal engagement and price responsiveness. |
| `TOTAL_SUPPORT_TICKETS_PRE_CUTOFF` | Integer | Support Friction | `fb Monthly Activity.SUPPORT_TICKETS` | $\sum \text{SUPPORT\_TICKETS}$ prior to cutoff. | Operational customer service contacts. High contact indicates store or delivery issues. |
| `COMPLAINT_RECORDED_PRE_CUTOFF` | Integer (0/1) | Support Friction | `fb Monthly Activity.COMPLAINT_FLAG` | `1 if max(COMPLAINT_FLAG) > 0 else 0` | Escalated customer dissatisfaction. Unresolved complaints strongly correlate with defection. |
| `HAS_NO_PRE_CUTOFF_ACTIVITY` | Integer (0/1) | Quality Flag | Activity Table Join | `1 if customer has 0 pre-cutoff activity rows else 0` | Flags the 119 members without pre-cutoff activity records (51 post-cutoff + 68 unrecorded). |
| `IS_LIMITED_PRE_CUTOFF_HISTORY` | Integer (0/1) | Quality Flag | Tenure and Activity | `1 if TENURE_DAYS < 90 or OBSERVED_MONTHS < 4 else 0` | Replaces end-of-observation flag without lookahead leakage. Identifies newer signups. |
| `VALIDATION_COHORT` | String | Model Validation | `SIGNUP_DATE` | `TRAIN` ($\le \text{2023-06-30}$), `VAL` ($\text{2023-07-01 to 2024-03-31}$), `EXCLUDED_POST_CUTOFF` ($> \text{2024-03-31}$) | **Validation Split Assignment.** Segments population for realistic out-of-cohort testing. |
| `CHURNED` | Integer (0/1) | **Target Variable** | `fb Churn Label.CHURNED` | Ground-truth outcome as of 2024-06-30. | **Supervised Learning Target.** Zero transactions in April–June 2024 despite prior history. |

---

## 3. Strict Target Leakage Protections Enforced

The following fields present in the raw workbook were **strictly excluded** from `phase2_modeling_dataset.csv`:
1. `LAST_PURCHASE_DATE`: Derived as of June 30, 2024. Excluded to prevent target leakage. Replaced by `RECENCY_MONTHS_PRE_CUTOFF`.
2. `MONTHS_OBSERVED`: Computed across the full 18-month window through June 2024. Replaced by `PRE_CUTOFF_OBSERVED_MONTHS` and `ELIGIBLE_PRE_CUTOFF_MONTHS`.
3. `INSUFFICIENT_HISTORY_FLAG`: Calculated at the end of the observation window. Replaced by `IS_LIMITED_PRE_CUTOFF_HISTORY`.
4. `OBSERVATION_END_DATE`: Constant metadata (`2024-06-30`). Excluded.
5. All monthly activity rows dated `2024-04-01`, `2024-05-01`, and `2024-06-01`: Excluded from all feature aggregates.
