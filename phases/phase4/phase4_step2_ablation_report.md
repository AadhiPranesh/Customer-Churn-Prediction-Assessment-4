# Phase 4 — Step 2: Feature Ablation Study Report (Revised)
**FreshBasket Loyalty Customer Churn Prediction Project (Propel Internship Assessment)**  
*Date: October 2026*

---

## Executive Summary

Phase 4 Step 2 performs a controlled **Feature Ablation Study** to systematically isolate and quantify how different categories of customer data contribute to churn prediction performance. By evaluating four cumulative feature subsets under identical model architectures, preprocessing pipelines, class weights, and validation splits, we determine which signals provide primary predictive power versus operational refinement.

### Key Step 2 Takeaways
1. **Demographics Show Little Useful Predictive Power in This Experiment:** Model Set 1 (Age, Gender, City, Tier, Price Tier) achieves a **PR-AUC of 0.1410**, which is virtually identical to the validation baseline churn prevalence ($14.12\%$), and a **ROC-AUC of 0.4807**, essentially indistinguishable from a random coin toss ($0.50$). In this dataset and experimental setup, static demographic attributes alone do not provide useful predictive discrimination for customer defection.
2. **Tenure History Without Purchase Recency Performs Poorly on This Validation Cohort:** Model Set 2 adds tenure days and observation months to demographics. Because the validation cohort consists of newer signups with shorter observation windows than the training cohort, the balanced decision tree ensemble flagged nearly every validation customer as at-risk based on tenure features alone. This resulted in **510 False Positives** out of 517 non-churners and collapsed accuracy to **14.95%**. Set 2 (Demographics + History) shows very low accuracy (about 0.15) at the 0.50 threshold because its probabilities are poorly separated and it predicts almost every customer as churn (recall 0.98, but only 7 true negatives), so PR-AUC and ROC-AUC are the fairer comparison for the ablation. This demonstrates that tenure history must be anchored by recent purchase recency to generalize across cohorts with differing tenure lengths.
3. **Purchase Behavior Drives the Primary Predictive Inflection:** Model Set 3 adds historical transaction volume, spending, basket value, coupon redemptions, and critical 90-day momentum (`TXNS_LAST_3M`, `SPEND_LAST_3M`, `RECENCY`). This produces a massive performance jump: **PR-AUC leaps from 0.1791 to 0.8290** (+0.6499), **ROC-AUC reaches 0.9327**, and **Accuracy reaches 91.03%**. Transaction velocity is the foundation of churn prediction in this retail loyalty program.
4. **Support and Digital Engagement Provide Critical Operational Refinement:** Model Set 4 (Full Feature Set) adds support ticket friction, complaints, and digital app/email engagement. This refinement increases **Recall from 75.29% to 82.35%** (identifying **+6 additional churners**, reducing missed defections from 21 to 15) and lifts **PR-AUC to 0.8499**.
5. **Non-Causal Overlap Note:** Feature groups overlap in the underlying customer behaviors they capture (e.g., active shoppers both purchase groceries and open digital emails). The measured performance gains reflect **incremental predictive utility within this model architecture**, not proof of causal mechanisms.

---

## 1. Feature Set Definitions & Strict Subset Verification

We verified that the feature sets satisfy a strict nested hierarchy: $\text{Set 1} \subset \text{Set 2} \subset \text{Set 3} \subset \text{Set 4}$.

```
                              Ablation Hierarchy
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Set 1: Demographics & Membership (6 features)                          │
  │ • AGE, AGE_IS_MISSING, GENDER, CITY, MEMBERSHIP_TIER,                  │
  │   HOME_STORE_PRICE_TIER                                                │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Set 2: Demographics + History (10 features)                            │
  │ • Set 1 + ELIGIBLE_PRE_CUTOFF_MONTHS, TENURE_DAYS_PRE_CUTOFF,          │
  │   PRE_CUTOFF_OBSERVED_MONTHS, IS_LIMITED_PRE_CUTOFF_HISTORY            │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Set 3: Purchase Behavior (22 features)                                 │
  │ • Set 2 + TOTAL_TXNS_PRE_CUTOFF, TOTAL_SPEND_PRE_CUTOFF,               │
  │   ACTIVE_MONTHS_PRE_CUTOFF, RECENCY_MONTHS_PRE_CUTOFF,                 │
  │   TXNS_LAST_3M, SPEND_LAST_3M, HAS_NO_PRE_CUTOFF_ACTIVITY,             │
  │   AVG_MONTHLY_TXNS_PRE_CUTOFF, AVG_MONTHLY_SPEND_PRE_CUTOFF,           │
  │   AVG_BASKET_VALUE_PRE_CUTOFF, NEVER_PURCHASED_PRE_CUTOFF,             │
  │   TOTAL_COUPONS_REDEEMED_PRE_CUTOFF                                    │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Set 4: Full Feature Set — Reference (28 features)                      │
  │ • Set 3 + TOTAL_SUPPORT_TICKETS_PRE_CUTOFF,                            │
  │   COMPLAINT_RECORDED_PRE_CUTOFF, TOTAL_APP_SESSIONS_PRE_CUTOFF,        │
  │   TOTAL_EMAILS_OPENED_PRE_CUTOFF, AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF, │
  │   AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF                                 │
  └────────────────────────────────────────────────────────────────────────┘
```

### Exact Breakdown of Features Added at Each Stage:
* **Added from Set 1 to Set 2 (4 features):**
  1. `ELIGIBLE_PRE_CUTOFF_MONTHS` (Exposure calendar months between signup and cutoff)
  2. `TENURE_DAYS_PRE_CUTOFF` (Days between signup and March 31, 2024)
  3. `PRE_CUTOFF_OBSERVED_MONTHS` (Distinct activity record months pre-cutoff)
  4. `IS_LIMITED_PRE_CUTOFF_HISTORY` (Indicator for signups with $<3$ months exposure)
* **Added from Set 2 to Set 3 (12 features):**
  1. `TOTAL_TXNS_PRE_CUTOFF` (Cumulative transactions pre-cutoff)
  2. `TOTAL_SPEND_PRE_CUTOFF` (Cumulative grocery spend pre-cutoff)
  3. `ACTIVE_MONTHS_PRE_CUTOFF` (Months with $\ge 1$ purchase)
  4. `RECENCY_MONTHS_PRE_CUTOFF` (Months since last pre-cutoff purchase)
  5. `TXNS_LAST_3M` (Transaction count during Jan–Mar 2024)
  6. `SPEND_LAST_3M` (Total spend during Jan–Mar 2024)
  7. `HAS_NO_PRE_CUTOFF_ACTIVITY` (Flag for zero transactions)
  8. `AVG_MONTHLY_TXNS_PRE_CUTOFF` (Transactions per eligible month)
  9. `AVG_MONTHLY_SPEND_PRE_CUTOFF` (Spend per eligible month)
  10. `AVG_BASKET_VALUE_PRE_CUTOFF` (Spend per transaction)
  11. `NEVER_PURCHASED_PRE_CUTOFF` (Flag for zero lifetime purchases)
  12. `TOTAL_COUPONS_REDEEMED_PRE_CUTOFF` (Promotional discount coupon redemption during checkout)
* **Added from Set 3 to Set 4 (6 features — Customer Support & Digital Engagement):**
  1. `TOTAL_SUPPORT_TICKETS_PRE_CUTOFF` — *Customer Support:* Cumulative support tickets submitted pre-cutoff.
  2. `COMPLAINT_RECORDED_PRE_CUTOFF` — *Customer Complaints:* Binary indicator for formal complaint recorded pre-cutoff.
  3. `TOTAL_APP_SESSIONS_PRE_CUTOFF` — *Digital App Activity:* Cumulative mobile app login sessions pre-cutoff.
  4. `AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF` — *Digital App Activity:* Average mobile app sessions per eligible month.
  5. `TOTAL_EMAILS_OPENED_PRE_CUTOFF` — *Email Engagement:* Cumulative marketing promotional emails opened pre-cutoff.
  6. `AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF` — *Email Engagement:* Average marketing emails opened per eligible month.

---

## 2. Comprehensive Ablation Metrics Table

All four models were trained on the exact training cohort ($N = 1,947$, 733 churned) and evaluated on the exact validation cohort ($N = 602$, 85 churned, 517 retained) using identical hyperparameters and the standard $0.50$ decision threshold:

| Model Variant | Features ($N$) | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | True Neg ($TN$) | False Pos ($FP$) | False Neg ($FN$) | True Pos ($TP$) | Total ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Set 1: Demographics Only** | 6 | 0.5482 | 0.1390 | 0.4235 | 0.2093 | **0.1410** | **0.4807** | 294 | 223 | 49 | 36 | **602** |
| **Set 2: Demog + History** | 10 | 0.1495 | 0.1400 | **0.9765** | 0.2448 | **0.1791** | **0.5915** | 7 | 510 | **2** | **83** | **602** |
| **Set 3: Purchase Behavior** | 22 | 0.9103 | 0.6598 | 0.7529 | 0.7033 | **0.8290** | **0.9327** | **484** | **33** | 21 | 64 | **602** |
| **Set 4: Full Set (Reference)**| **28** | **0.9169** | **0.6667** | 0.8235 | **0.7368** | **0.8499** | **0.9446** | 482 | 35 | 15 | 70 | **602** |

### Confusion Matrix Reconciliation:
* **Set 1:** Retained ($294 + 223 = 517$), Churned ($49 + 36 = 85$), Total = **602**.
* **Set 2:** Retained ($7 + 510 = 517$), Churned ($2 + 83 = 85$), Total = **602**.
* **Set 3:** Retained ($484 + 33 = 517$), Churned ($21 + 64 = 85$), Total = **602**.
* **Set 4:** Retained ($482 + 35 = 517$), Churned ($15 + 70 = 85$), Total = **602**.

> [!NOTE]
> **Minority Class Metric Focus:**  
> Because actual churners represent only **$14.12\%$** of the validation cohort (85 out of 602), standard Accuracy is misleading. **PR-AUC (Precision-Recall Area Under the Curve)** and **Recall** are the primary analytical metrics:
> * A random baseline classifier on this cohort achieves a PR-AUC equal to the positive prevalence (**0.1412**).
> * Model Set 1 achieves **0.1410**, demonstrating that static demographic and membership tiers offer no useful predictive lift over naive guessing in this experiment.

---

## 3. Delta Comparison Relative to Full Feature Set (Reference)

To clearly illustrate the marginal contribution of each data block, we calculate the performance difference for each smaller feature set relative to the full reference model ($\Delta = \text{Metric}_{\text{Subset}} - \text{Metric}_{\text{Full}}$):

| Feature Set | Features | $\Delta$ PR-AUC | $\Delta$ Recall | $\Delta$ Precision | $\Delta$ F1-Score | $\Delta$ ROC-AUC | Missed Churners ($FN$) | False Alarms ($FP$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Set 1: Demographics Only** | 6 | **-0.7089** | **-0.4000** | -0.5277 | -0.5275 | -0.4639 | 49 (+34) | 223 (+188) |
| **Set 2: Demographics + History** | 10 | **-0.6708** | +0.1530 | -0.5267 | -0.4920 | -0.3531 | 2 (-13) | **510 (+475)** |
| **Set 3: Purchase Behavior** | 22 | **-0.0209** | **-0.0706** | -0.0069 | -0.0335 | -0.0119 | 21 (+6) | 33 (-2) |
| **Set 4: Full Feature Set (Reference)**| 28 | **0.0000** | **0.0000** | 0.0000 | 0.0000 | 0.0000 | **15** | **35** |

---

## 4. In-Depth Interpretation of Feature Group Contributions

### 4.1 Set 1: Demographics and Membership Tiers
* **Findings:**
  * PR-AUC is **0.1410**, essentially identical to the random baseline prevalence ($0.1412$).
  * ROC-AUC is **0.4807**, indicating performance indistinguishable from a coin toss.
  * Precision is **13.90%**, meaning approximately 6 out of 7 customers flagged by demographic rules alone are false alarms.
* **Cautious Plain-Language Explanation:**  
  "In this experiment, customer attributes such as age, gender, city, and loyalty tier provided little useful predictive power for churn. While demographics may define high-level marketing segments, members of all ages, genders, and tiers churn when their personal shopping patterns change. Demographics alone are insufficient for predicting defection in this dataset."

---

### 4.2 Set 2: Demographics Plus History/Tenure
* **Findings:**
  * Recall is high at **97.65%** (83 out of 85 churners caught), but Accuracy collapses to **14.95%** and Precision remains stuck at **14.00%**.
  * The model generated **510 False Positives** out of 517 non-churners.
* **Why This Occurred on This Validation Cohort:**  
  In the training cohort, customers enrolled earlier had longer observation windows and a higher historical churn rate ($37.65\%$). The decision trees split heavily on tenure length to partition older high-churn customers. When applied to the validation cohort—where *all* customers have shorter tenure—the balanced tree ensemble flagged virtually the entire population as at-risk based on tenure features alone.  
* **Cautious Plain-Language Explanation:**  
  "On this validation cohort, tenure and exposure history performed poorly when evaluated without purchase recency. Because newer cohorts naturally have shorter tenures, the model misapplied historical tenure correlations, treating almost all new customers as defectors. This illustrates that tenure features must be combined with recent shopping activity to generalize across cohorts with differing enrollment dates."

---

### 4.3 Set 3: Purchase Behavior
* **Findings:**
  * PR-AUC undergoes an enormous leap from **0.1791 to 0.8290** (+0.6499).
  * ROC-AUC climbs from **0.5915 to 0.9327** (+0.3412).
  * Accuracy jumps to **91.03%**, with False Positives dropping from 510 down to **33**.
  * Recall reaches **75.29%** (64 out of 85 churners caught).
* **Plain-Language Explanation:**  
  "Transaction velocity is the single most important block of information in churn modeling. Incorporating whether a customer shopped in the last 90 days, how much they spent, and how many months have passed since their last trip drives the core discrimination. Behavioral data does the heavy lifting in this model."

---

### 4.4 Set 4: Full Feature Set (Support & Digital Engagement Refinement)
* **Findings:**
  * Recall increases by **+7.06 percentage points** (from 75.29% to **82.35%**).
  * True Positives increase from **64 to 70**, successfully capturing **6 additional churners** who had maintained moderate transaction volume but were actively experiencing service friction.
  * Missed churners ($FN$) drop from **21 to 15** (a **28.6% reduction in missed defections**).
  * PR-AUC increases from **0.8290 to 0.8499** (+0.0209); ROC-AUC reaches **0.9446**.
* **Plain-Language Explanation:**  
  "While purchase history flags customers who have already stopped buying, customer service friction and digital engagement data identify borderline members who are frustrated or disengaging. Adding support tickets and app/email touches captured 6 additional churners who would have slipped through a purely transaction-based model."

---

## 5. Non-Causal Overlap and Methodological Limitations

> [!WARNING]
> **Correlated Feature Overlap & Causation Disclaimer:**  
> 1. **Collinear Redundancy:** Feature categories do not operate in isolation. Customers who actively purchase groceries also tend to use the mobile app, redeem coupons, and open emails. Because of this natural collinearity, the marginal gain observed between Set 3 and Set 4 (+0.0209 PR-AUC) reflects the *incremental* residual variance explained after purchase behavior is already accounted for.
> 2. **Predictive Utility vs. Intervention Efficacy:** Finding that support tickets and digital engagement improve churn classification does **not** prove that manipulating these features will causally alter customer retention. For example, while low app usage predicts churn, forcing app notifications will not fix underlying merchandise or pricing dissatisfaction.

---

## 6. Deliverables Created & Verification Status

All Step 2 code, data tables, and charts were generated and verified in the workspace root and [`phase_4_outputs/`](../../phase_4_outputs):

1. [phase4_step2_ablation_study.py](../../phase4_step2_ablation_study.py): Complete, reproducible ablation script running all 4 models under identical Random Forest settings.
2. [phase4_step2_ablation_results.csv](../../phase4_step2_ablation_results.csv): Metrics table recording Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC, and full confusion matrix values for all four variants.
3. [phase4_step2_ablation_deltas.csv](../../phase4_step2_ablation_deltas.csv): Formatted table of performance deltas relative to Set 4.
4. [phase_4_outputs/charts/phase4_ablation_comparison.png](../../phase_4_outputs/charts/phase4_ablation_comparison.png): Two-panel visualization comparing PR-AUC/ROC-AUC and Recall/Precision/F1 across the 4 subsets.
5. [phase4_step2_ablation_report.md](../../phase4_step2_ablation_report.md): This comprehensive report.

---
*Phase 4 — Step 2 targeted quality review is complete. As instructed, execution has stopped before Step 3 (Retention Intervention Simulation).*
