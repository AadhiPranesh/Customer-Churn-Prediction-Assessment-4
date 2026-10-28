# Phase 1: Exploratory Data Analysis (EDA) Report

**Project:** FreshBasket Loyalty Program Customer Churn Prediction  
**Assessment:** Propel Customer Churn Assessment  
 
**Date:** October 2026  
**Scope:** Phase 1 (EDA and Behavioral Patterns; No Predictive Modeling)

---

## 1. Executive Summary of Exploratory Findings

Exploratory Data Analysis (EDA) was performed across the cleaned 2,600 loyalty members and their historical monthly shopping records. To preserve scientific validity and prevent lookahead bias, **all behavioral, spend, and friction comparisons between churners and non-churners were conducted strictly on pre-cutoff data (January 1, 2023 through March 31, 2024)**. April–June 2024 activity was examined solely to verify the ground-truth target definition.

### Top Analytical Takeaways:
1. **Target Distribution:** The loyalty program exhibits a **31.46% churn rate** (818 churned vs. 1,782 retained members), reflecting an approximate 1:2.17 class imbalance.
2. **Behavioral Divergence:** Churners do not leave abruptly without warning. During the 15 months *before* the churn window, members who eventually churned averaged **46.1% lower monthly spend** (\$75.48/mo vs. \$140.15/mo) and **45.5% fewer shopping visits** (2.36 vs. 4.33 transactions/mo) compared to retained members.
3. **Digital Disengagement:** Churned members recorded **50% fewer mobile app sessions** (1.08 vs. 2.14 sessions/mo) and opened 35.7% fewer marketing emails (0.67 vs. 1.05 emails/mo) in the pre-cutoff period.
4. **Support and Friction Signals:** Churned members logged **43.5% more customer support tickets** (1.85 vs. 1.29 tickets) and were significantly more likely to have registered a formal complaint (38.2% vs. 30.7%) prior to cutoff.
5. **Demographic Weakness:** Member age is virtually identical between retained members (mean 45.9 years) and churned members (mean 46.7 years), indicating that demographic variables alone will provide weak churn discrimination compared to behavioral engagement signals.

---

## 2. Visual and Statistical Findings

### 2.1 Target Churn Distribution
![Churn Distribution](../../phase_1_outputs/charts/01_churn_distribution.png)

* **Observed Counts:** Retained ($y=0$): **1,782 members ($68.54\%$)**; Churned ($y=1$): **818 members ($31.46\%$)**.
* **Analytical Interpretation:** While non-churners constitute the majority, the 31.5% churn rate represents a substantial commercial loss (nearly one-third of the loyalty base). From a modeling perspective, an uncalibrated classifier that defaults to predicting the majority class would achieve 68.5% accuracy while catching zero churners. This confirms why **PR-AUC and F1-score** must supersede Accuracy during Phase 3 evaluation.

---

### 2.2 Missing Values Audit
![Missing Values Summary](../../phase_1_outputs/charts/02_missing_values_summary.png)

* **Observed Missingness:**
  - `HOME_STORE_PRICE_TIER`: 130 records missing ($4.98\%$).
  - `LAST_PURCHASE_DATE`: 120 records missing ($4.61\%$). (Corresponds strictly to inactive signups who never made a purchase).
  - `AGE`: 78 records missing ($2.99\%$).
  - `GENDER`: 52 records missing ($1.99\%$).
  - All other fields in `fb Customer Profile` and all fields in `fb Monthly Activity` have zero missing values ($100\%$ complete).
* **Analytical Interpretation:** Missingness is relatively low ($< 5\%$ across all fields). Categorical missingness was cleanly encapsulated as an explicit `'Unknown'` category, allowing models to evaluate whether non-reporting is itself predictive of churn risk.

---

### 2.3 Macro-Level Monthly Spending and Transaction Trends
![Monthly Trends](../../phase_1_outputs/charts/03_monthly_trends_overall.png)

* **Observed Trend:** Aggregate monthly spending across the loyalty program averaged \$180,000–\$210,000 per month throughout 2023, accompanied by ~6,000–7,000 monthly transactions.
* **Target Window Impact:** Following the **March 31, 2024 Prediction Cutoff**, aggregate spend and transactions drop noticeably during April–June 2024. This drop reflects the quiet defection of the 818 churned members who made zero purchases during this final quarter.
* **Lookahead Warning:** This chart visually underscores why April–June 2024 data must remain sealed during feature engineering. Using any aggregate or customer-level figures from this period would leak the outcome directly into model inputs.

---

### 2.4 Customer Age Distribution
![Age Distribution](../../phase_1_outputs/charts/04_customer_age_distribution.png)

| Churn Status | Count (Non-Null) | Mean Age | Median Age | Std Dev | Min Age | Max Age | IQR (25%–75%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Retained (0)** | 1,728 | 45.88 yrs | 46.0 yrs | 16.46 | 18.0 | 74.0 | 31.0 – 60.0 yrs |
| **Churned (1)** | 794 | 46.73 yrs | 47.0 yrs | 16.17 | 18.0 | 74.0 | 33.0 – 60.8 yrs |

* **Analytical Interpretation:** The age distribution is broad (18 to 74 years) and uniformly distributed across both cohorts. The mean difference between churners and non-churners is less than **0.85 years** ($p > 0.20$). Age alone cannot differentiate at-risk shoppers from loyal customers.

---

### 2.5 Churn Distribution Across Customer Categories
![Churn by Categories](../../phase_1_outputs/charts/05_churn_by_categories.png)

#### A. Membership Tier
* `Silver` Tier: **32.29% Churn** (476 churned / 1,474 members).
* `Platinum` Tier: **31.13% Churn** (94 churned / 302 members).
* `Gold` Tier: **30.10% Churn** (248 churned / 824 members).
* *Interpretation:* Churn prevalence is surprisingly consistent across loyalty tiers (~30%–32%). While Silver members have a slightly higher attrition rate (+2.2% vs Gold), even high-tier Platinum members experience ~31% churn, indicating that tier benefits alone do not insulate FreshBasket from customer attrition.

#### B. Home Store Price Tier
* `Value` Stores: **34.15% Churn** (305 churned / 893 members).
* `Mainstream` Stores: **30.63% Churn** (355 churned / 1,159 members).
* `Unknown` Stores: **29.23% Churn** (38 churned / 130 members).
* `Upscale` Stores: **28.71% Churn** (120 churned / 418 members).
* *Interpretation:* Members shopping primarily at `Value` price-tier stores exhibit a noticeably higher churn rate ($34.2\%$) than those at `Upscale` stores ($28.7\%$). This observed pattern suggests higher price sensitivity and competitor deal-seeking among value-oriented grocery shoppers.

#### C. Member Gender
* `Unknown`: **40.38% Churn** (21 / 52 members).
* `Female (F)`: **32.70% Churn** (447 / 1,367 members).
* `Male (M)`: **29.77% Churn** (323 / 1,085 members).
* `Other`: **28.13% Churn** (27 / 96 members).
* *Interpretation:* Members with unrecorded gender exhibit the highest churn rate (40.4%), which may correlate with lower initial engagement during loyalty signup.

#### D. Geographic Metropolitan Area (10 Cities)
* Top Churn Cities: **Austin (37.59%)**, **Columbus (36.19%)**, **Nashville (33.33%)**.
* Moderate Churn Cities: **Boise (31.98%)**, **Tulsa (29.83%)**, **Raleigh (29.79%)**, **Madison (29.69%)**.
* Lowest Churn Cities: **Spokane (28.97%)**, **Denver (28.79%)**, **Portland (28.33%)**.
* *Interpretation:* A 9.3 percentage point spread exists between Austin ($37.6\%$) and Portland ($28.3\%$). Local competitive pressures (e.g., competing grocery chains opening in Austin) or store operational differences may explain regional variance.

---

### 2.6 Historical Spending and Purchase Frequency (Pre-Cutoff)
![Historical Spending Behavior](../../phase_1_outputs/charts/06_historical_spending_behavior.png)

To rigorously evaluate behavioral predictors, we aggregated monthly activity strictly over the **15-month pre-cutoff period (January 2023 – March 2024)**:

| Metric (Pre-Cutoff Monthly Average) | Retained Members ($y=0$) | Churned Members ($y=1$) | Relative Difference |
| :--- | :--- | :--- | :--- |
| **Mean Monthly Spend** | **\$140.15** (Median \$136.45) | **\$75.48** (Median \$67.26) | **-46.1% lower spend** |
| **Monthly Transactions** | **4.33 txns** (Median 4.38) | **2.36 txns** (Median 2.15) | **-45.5% fewer visits** |
| **Average Basket Value** | **\$29.09** (Median \$30.14) | **\$25.00** (Median \$25.04) | **-14.1% smaller basket** |

* **Analytical Interpretation:** Churners were already disengaging months prior to the April 2024 cutoff. They shopped less than half as often and spent substantially less per visit. This confirms Hypothesis 1 and Hypothesis 2: visit velocity and basket size are primary candidates for early-warning feature engineering.

---

### 2.7 Support Friction and Digital Engagement (Pre-Cutoff)
![Engagement and Friction Comparison](../../phase_1_outputs/charts/07_engagement_and_friction_comparison.png)

| Engagement / Friction Metric | Retained Members ($y=0$) | Churned Members ($y=1$) | Observed Pattern |
| :--- | :--- | :--- | :--- |
| **Members with Formal Complaint** | **30.69%** | **38.24%** | Churners logged **+24.6% more complaints** |
| **Mean Pre-Cutoff Support Tickets** | **1.29 tickets** (Median 1.0) | **1.85 tickets** (Median 2.0) | Churners logged **+43.4% more support tickets** |
| **Mean App Sessions per Month** | **2.14 sessions** (Median 2.08) | **1.08 sessions** (Median 1.00) | Churners logged **-49.5% fewer app sessions** |
| **Mean Emails Opened per Month** | **1.05 emails** (Median 1.09) | **0.67 emails** (Median 0.50) | Churners opened **-36.2% fewer marketing emails** |

* **Analytical Interpretation:** Friction and disengagement are strongly co-occurring with churn. Members who logged formal complaints (`COMPLAINT_FLAG = 1`) had a significantly higher propensity to churn. Concurrently, digital engagement (app sessions and email opens) was halved among future churners.
* **Causal Disclaimer:** These findings demonstrate strong statistical association, but **do not prove causality**. For example, a customer may submit a complaint because of poor store service and subsequently churn, or an already-disaffected customer may simply be more prone to complaining before their quiet departure.

---

## 3. Methodological Summary for Phase 2 Transition

Phase 1 EDA confirms that:
1. Historical transaction frequency, spending erosion, app disengagement, and support complaints provide the clearest empirical signals of churn.
2. Static demographics (age, gender, tier) show minimal standalone variance across churn classes.
3. Feature engineering in Phase 2 must focus heavily on **velocity metrics** (e.g., Q1 2024 vs. Q3 2023 transaction ratio, 3-month spend slope, and days since last purchase prior to March 31, 2024).
