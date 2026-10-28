# Phase 4 — Step 1: Diagnostic Driver Attribution and Segment Risk Profiling Report (Revised)
**FreshBasket Loyalty Customer Churn Prediction Project (Propel Internship Assessment)**  
*Date: October 2026*

---

## Executive Summary

Phase 4 transitions the project from predictive validation into **diagnostic interpretation, feature attribution, and customer risk profiling**. As required by the assessment brief, **Step 1** focuses strictly on:
1. Identifying and explaining the primary drivers of churn across both the Random Forest model and the Logistic Regression baseline.
2. Clarifying the fundamental methodological distinction between **predictive importance** and **causation**.
3. Profiling churn patterns across customer segments (Membership Tiers, Home Store Price Tiers, and Cities) while evaluating sample size reliability.
4. Using SHAP TreeExplainer values on the holdout validation cohort ($N = 602$) to provide global and customer-level diagnostic attributions.

### Methodological & Model Governance Verification
* **Model Provenance & Reuse:** Step 1 directly reuses the exact serialized Phase 3 Random Forest pipeline ([`phase3_random_forest_pipeline.joblib`](../../phase3_random_forest_pipeline.joblib)), which contains the fitted preprocessing pipeline and a `RandomForestClassifier` with 200 trees (`max_depth=6, min_samples_leaf=10, class_weight='balanced', random_state=42`). Its output probabilities match [`phase3_validation_predictions.csv`](../../phase3_validation_predictions.csv) to 4 decimal places (maximum absolute difference $< 0.00005$). No separate or altered model was introduced.
* **SHAP Mathematical Verification:** SHAP TreeExplainer was evaluated on the validation cohort ($N = 602$). We verified that:
  - SHAP values explain **Class 1 (Churned)** on the probability scale.
  - The ensemble base expected value for the churn class is **$0.4987$** (reflecting balanced class weighting).
  - For every customer $i$, $\text{Base Value} + \sum_j \text{SHAP}_{i,j} = \hat{P}_i(\text{churn})$ exactly (maximum additivity error $< 10^{-8}$).
* **Non-Causal Interpretations:** All model coefficients and importance scores are interpreted strictly as **observational statistical associations**. All qualitative explanations regarding retail shopping habits are explicitly labeled as **untested business hypotheses**.

---

## 1. Top Predictors of Churn in Plain, Interview-Friendly Language

Across both the non-linear Random Forest and the linear Logistic Regression baseline, the same behavioral signals consistently dominate predictive power:

```
                            Feature Importance Comparison
  ┌─────────────────────────────────────┬──────────────────┬─────────────────┬─────────────────────────────┐
  │ Feature Name                        │ RF Gini Imp (MDI)│ SHAP Mean |SHAP|│ LR Standardized Coeff (1σ) │
  ├─────────────────────────────────────┼──────────────────┼─────────────────┼─────────────────────────────┤
  │ TXNS_LAST_3M                        │   17.41% (#1)    │   0.0979 (#2)   │ β = -3.0280 (OR = 0.0484)   │
  │ RECENCY_MONTHS_PRE_CUTOFF           │   16.86% (#2)    │   0.0966 (#3)   │ β = +0.6130 (OR = 1.8460)   │
  │ SPEND_LAST_3M                       │   16.49% (#3)    │   0.1052 (#1)   │ β = -2.0986 (OR = 0.1226)   │
  │ AVG_MONTHLY_TXNS_PRE_CUTOFF         │    6.68% (#4)    │   0.0270 (#5)   │ β = +0.8643 (OR = 2.3732)   │
  │ ACTIVE_MONTHS_PRE_CUTOFF            │    6.51% (#5)    │   0.0318 (#4)   │ β = -0.0043 (OR = 0.9957)   │
  │ AVG_MONTHLY_SPEND_PRE_CUTOFF        │    5.70% (#6)    │   0.0260 (#7)   │ β = +0.4644 (OR = 1.5910)   │
  │ TOTAL_TXNS_PRE_CUTOFF               │    4.99% (#7)    │   0.0261 (#6)   │ β = +0.4848 (OR = 1.6238)   │
  │ AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF │    4.79% (#8)    │   0.0213 (#8)   │ β = -0.3719 (OR = 0.6894)   │
  │ TOTAL_SUPPORT_TICKETS_PRE_CUTOFF    │    1.42%         │   0.0078        │ β = +0.6281 (OR = 1.8740)   │
  │ AVG_MONTHLY_EMAILS_OPENED           │    1.56%         │   0.0051        │ β = -0.6959 (OR = 0.4986)   │
  └─────────────────────────────────────┴──────────────────┴─────────────────┴─────────────────────────────┘
```

### Top Predictor Explanations (Per One Standard Deviation for Numeric Features)

1. **Recent Transaction Momentum (`TXNS_LAST_3M`):**
   * *Metric Summary:* RF Gini = 17.41% (#1); SHAP = 0.0979 (#2); LR $\beta = -3.0280$ ($\text{Odds Ratio} = 0.0484$).
   * *Training Scale:* Standard deviation $\sigma = 8.14$ transactions.
   * *Statistical Interpretation:* Holding other standardized features constant, each one-standard-deviation increase in transactions during the last 3 months (+8.14 transactions) is associated with an estimated **95.2% decrease in the odds of churn** ($\text{Odds Ratio} = 0.0484$).
   * *Plain Language:* "Customers who visited FreshBasket frequently during the final 90 days before the cutoff are statistically the least likely to defect. A sudden halt in visits is the primary observational signal of impending defection."

2. **Dormancy Recency (`RECENCY_MONTHS_PRE_CUTOFF`):**
   * *Metric Summary:* RF Gini = 16.86% (#2); SHAP = 0.0966 (#3); LR $\beta = +0.6130$ ($\text{Odds Ratio} = 1.8460$).
   * *Training Scale:* Standard deviation $\sigma = 20.84$ months (reflecting the 99-month fill for non-purchasers).
   * *Statistical Interpretation:* Holding other standardized features constant, each one-standard-deviation increase in recency is associated with an estimated **84.6% increase in the odds of churn** ($\text{Odds Ratio} = 1.8460$).
   * *Plain Language:* "As the number of months since a customer's last purchase increases, their statistical likelihood of remaining an active member drops significantly."

3. **Recent Spend Momentum (`SPEND_LAST_3M`):**
   * *Metric Summary:* RF Gini = 16.49% (#3); SHAP = 0.1052 (#1); LR $\beta = -2.0986$ ($\text{Odds Ratio} = 0.1226$).
   * *Training Scale:* Standard deviation $\sigma = \$294.70$.
   * *Statistical Interpretation:* Holding other standardized features constant, each one-standard-deviation increase in recent spend (+\$294.70 in Q1) is associated with an estimated **87.7% decrease in the odds of churn** ($\text{Odds Ratio} = 0.1226$).
   * *Plain Language:* "Customers who maintained active grocery spending in the 90 days leading up to the cutoff date are strongly correlated with continued loyalty."

4. **Customer Support Interaction Friction (`TOTAL_SUPPORT_TICKETS_PRE_CUTOFF`):**
   * *Metric Summary:* LR $\beta = +0.6281$ ($\text{Odds Ratio} = 1.8740$).
   * *Training Scale:* Standard deviation $\sigma = 1.37$ support tickets.
   * *Statistical Interpretation:* Holding other standardized features constant, each one-standard-deviation increase in support tickets (+1.37 tickets) is associated with an estimated **87.4% increase in the odds of churn** ($\text{Odds Ratio} = 1.8740$).
   * *Plain Language:* "Members with higher numbers of support tickets correlate with higher churn rates in historical data. We hypothesize that frequent support contacts reflect unresolved service or product friction."

5. **Digital Engagement (`AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF` & `AVG_MONTHLY_APP_SESSIONS`):**
   * *Metric Summary:* Emails LR $\beta = -0.6959$ ($\text{Odds Ratio} = 0.4986, \sigma = 0.61$ emails/mo); App RF Gini = 4.79% (#8, $\beta = -0.3719, \text{Odds Ratio} = 0.6894, \sigma = 1.54$ sessions/mo).
   * *Statistical Interpretation:* Each one-standard-deviation increase in average monthly emails opened (+0.61 emails/mo) is associated with an estimated **50.1% decrease in the odds of churn** ($\text{Odds Ratio} = 0.4986$).
   * *Plain Language:* "Members who regularly open monthly emails and interact with the mobile app have lower churn odds, statistically indicating stronger brand engagement."

---

## 2. Predictive Importance vs. Causation (Interview Preparation)

> [!IMPORTANT]
> **Key Methodological Principle:**  
> A feature importance score (MDI), SHAP value, or regression coefficient describes **statistical association and predictive utility within historical data**, not a **causal mechanism**.

### 1. What the Metrics Actually Measure
* **Random Forest Gini Impurity (MDI):** Measures how frequently and effectively a feature was selected to split tree nodes to reduce classification impurity on the training cohort. (Caveat: MDI can favor continuous features with many candidate thresholds over categorical variables).
* **SHAP Values (TreeExplainer):** Measures the fair additive contribution of a feature value to shifting an individual customer's predicted probability away from the base population average ($0.4987$).
* **Standardized Logistic Regression Coefficients:** Quantify the observational change in log-odds associated with a $1\sigma$ change in a feature, holding other modeled predictors constant.

### 2. Untested Business Hypotheses vs. Causal Facts
To maintain analytical rigor, retail explanations must be explicitly categorized as **untested hypotheses**:

1. **Customer Support Tickets:**
   * *Observational Finding:* Customers with more tickets have higher churn rates ($\text{Odds Ratio} = 1.874$).
   * *Untested Business Hypothesis:* We hypothesize that support tickets reflect operational dissatisfaction (e.g., delivery delays, billing errors, or damaged goods).
   * *Why It Is Not Causal:* Attempting to "lower churn" by eliminating support channels or discouraging ticket submissions would not reduce churn; it would hide customer complaints and likely accelerate defection.
2. **Email Engagement:**
   * *Observational Finding:* Customers opening more monthly emails have lower churn rates ($\text{Odds Ratio} = 0.499$).
   * *Untested Business Hypothesis:* We hypothesize that opening monthly emails reflects customer interest in weekly grocery planning and specials.
   * *Why It Is Not Causal:* Artificially spamming disengaged customers with additional emails will not turn them into loyal shoppers and could increase unsubscriptions.
3. **Value Store Shopper Volatility:**
   * *Observational Finding:* The model shows lower recall on Value store shoppers (66.7% vs. 90.2% on Mainstream).
   * *Untested Business Hypothesis:* We hypothesize that budget-conscious shoppers may shop across multiple retail stores depending on sales and circulars, creating irregular purchase gaps. This hypothesis is not measured directly by the dataset (which records no competitor activity).

---

## 3. Descriptive Segment Risk Profiling ($N = 602$, Fully Reconciled)

### 3.1 Membership Tier Breakdown ($N = 602$)

| Membership Tier | Customers ($N$) | Churned ($N$) | Actual Churn Rate | RF Mean Prob | RF Predicted Churners | RF Recall | RF Precision | Sample Reliability Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Silver** | 348 | 45 | 12.93% | 30.35% | 56 (16.09%) | 80.0% (36/45) | 64.3% (36/56) | Sufficient Sample Size ($N = 348$) |
| **Gold** | 191 | 26 | 13.61% | 30.65% | 36 (18.85%) | 92.3% (24/26) | 66.7% (24/36) | Sufficient Sample Size ($N = 191$) |
| **Platinum** | 63 | 14 | 22.22% | 29.52% | 13 (20.63%) | 71.4% (10/14) | 76.9% (10/13) | **Small Sample ($N = 63$, 14 churners)** |
| **Cohort Total**| **602** | **85** | **14.12%** | **30.36%** | **105 (17.44%)**| **82.4% (70/85)**| **66.7% (70/105)**| **100% Reconciled** |

#### Subgroup Sample Size Limitation (Platinum):
* Platinum members show the highest observed churn rate ($22.22\%$). However, because the segment contains **only 63 customers and 14 churners**, the $95\%$ binomial confidence interval around the churn rate is wide ($[12.6\%, 34.8\%]$). A change in outcome for just 2 customers alters recall by $\pm 14.3\%$. Therefore, while Platinum churn carries high commercial revenue risk per customer, claims about Platinum tier dynamics must be qualified by this sample size limitation.

---

### 3.2 Home Store Price Tier Breakdown (All 602 Customers Reconciled)

| Store Price Tier | Customers ($N$) | Churned ($N$) | Actual Churn Rate | RF Mean Prob | RF Predicted Churners | RF Recall | RF Precision | Sample Reliability Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mainstream** | 281 | 41 | 14.59% | 30.92% | 57 (20.28%) | 90.2% (37/41) | 64.9% (37/57) | Sufficient Sample Size ($N = 281$) |
| **Value** | 198 | 27 | 13.64% | 29.71% | 29 (14.65%) | 66.7% (18/27) | 62.1% (18/29) | Sufficient Sample Size ($N = 198$) |
| **Upscale** | 100 | 17 | 17.00% | 33.58% | 19 (19.00%) | 88.2% (15/17) | 78.9% (15/19) | Moderate Sample Size ($N = 100$) |
| **Unknown** | 23 | 0 | 0.00% | 15.06% | 0 (0.00%) | N/A ($0/0$) | N/A ($0/0$) | **Very Small Sample ($N = 23$, 0 churners)** |
| **Cohort Total**| **602** | **85** | **14.12%** | **30.36%** | **105 (17.44%)**| **82.4% (70/85)**| **66.7% (70/105)**| **100% Reconciled** |

#### Reconciliation of the 23 Unknown Store Tier Customers:
* Exactly 23 validation customers had missing store tier metadata in the raw profile table. Zero of these 23 customers churned ($y = 0$). The model correctly assigned all 23 a low churn probability (mean $= 15.06\%$, below the $0.50$ threshold), resulting in $100\%$ true negative classifications ($TN = 23, FP = 0$).

---

### 3.3 City-Level Breakdown ($N = 602$, 10 Cities)

| City | Customers ($N$) | Churned ($N$) | Actual Churn Rate | RF Mean Prob | RF Predicted Churners | RF Recall | RF Precision | Sample Reliability Warning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Austin** | 64 | 13 | 20.31% | 34.03% | 14 | 84.6% (11/13) | 78.6% (11/14) | Small Subgroup ($N=64$, 13 churners) |
| **Madison** | 62 | 10 | 16.13% | 30.90% | 13 | 90.0% (9/10) | 69.2% (9/13) | Small Subgroup ($N=62$, 10 churners) |
| **Denver** | 63 | 10 | 15.87% | 31.21% | 11 | 80.0% (8/10) | 72.7% (8/11) | Small Subgroup ($N=63$, 10 churners) |
| **Columbus** | 58 | 9 | 15.52% | 32.92% | 11 | 88.9% (8/9) | 72.7% (8/11) | Small Subgroup ($N=58$, 9 churners) |
| **Tulsa** | 71 | 11 | 15.49% | 30.55% | 14 | 81.8% (9/11) | 64.3% (9/14) | Small Subgroup ($N=71$, 11 churners) |
| **Spokane** | 50 | 6 | 12.00% | 30.63% | 8 | 83.3% (5/6) | 62.5% (5/8) | Small Subgroup ($N=50$, 6 churners) |
| **Boise** | 60 | 7 | 11.67% | 25.53% | 8 | 85.7% (6/7) | 75.0% (6/8) | Small Subgroup ($N=60$, 7 churners) |
| **Nashville** | 43 | 5 | 11.63% | 28.30% | 5 | 80.0% (4/5) | 80.0% (4/5) | Small Subgroup ($N=43$, 5 churners) |
| **Portland** | 70 | 8 | 11.43% | 30.04% | 12 | 62.5% (5/8) | 41.7% (5/12) | Small Subgroup ($N=70$, 8 churners) |
| **Raleigh** | 61 | 6 | 9.84% | 28.78% | 9 | 83.3% (5/6) | 55.6% (5/9) | Small Subgroup ($N=61$, 6 churners) |
| **Cohort Sum**| **602** | **85** | **14.12%** | **30.36%** | **105** | **82.4%** | **66.7%** | **100% Reconciled** |

> [!WARNING]
> **Sample Size Limitations for Geographic Data:**  
> The 10 cities each contain between **43 and 71 validation customers**, with only **5 to 13 actual churners per city**.
> * The observed rate difference between Austin ($20.3\%$) and Raleigh ($9.8\%$) has overlapping $95\%$ binomial confidence intervals (Austin: $[11.3\%, 32.2\%]$ vs. Raleigh: $[3.7\%, 20.2\%]$).
> * In both models, one-hot encoded city features carry very low feature importances and small regression coefficients.
> * **Conclusion:** Geographic differences in this dataset cannot be confirmed as statistically meaningful drivers of churn. Business retention strategies should focus on behavioral defection signals rather than city-level adjustments.

---

## 4. Individual Customer Case Studies (Verified SHAP Attributions)

We examined three individual validation cohort members to illustrate how the model computes predictions using TreeExplainer SHAP values. For each customer, the model starts at the baseline expected value of **$0.4987$** and sums the individual feature SHAP values to reach the exact predicted churn probability ($\text{Base Value} + \sum \text{SHAP} = \hat{P}$):

### Case Study 1: High-Confidence Churner (True Positive)
* **Customer ID:** `500333` | **Tier:** Gold | **Store:** Mainstream | **City:** Tulsa
* **Actual Outcome:** **Churned ($y = 1$)**
* **Model Output:** Predicted Churn Probability = **98.83%** (LR Probability = **99.53%**)
* **SHAP Verification:** $\text{Base Value } (0.4987) + \sum \text{SHAP } (+0.4895) = \mathbf{0.9883}$
* **Customer Profile:**
  - `TXNS_LAST_3M` = **0** transactions
  - `SPEND_LAST_3M` = **$0.00**
  - `RECENCY_MONTHS_PRE_CUTOFF` = **5 months**
  - `TOTAL_SUPPORT_TICKETS_PRE_CUTOFF` = **4 tickets**
* **Top Features Driving Risk UP (Towards Churn):**
  - `TXNS_LAST_3M`: **$+0.0970$**
  - `RECENCY_MONTHS_PRE_CUTOFF`: **$+0.0913$**
  - `SPEND_LAST_3M`: **$+0.0866$**
* **Top Features Driving Risk DOWN (Towards Retention):**
  - `ELIGIBLE_PRE_CUTOFF_MONTHS`: $-0.0058$
  - `TENURE_DAYS_PRE_CUTOFF`: $-0.0052$
* **Diagnostic Summary:** Zero grocery visits and \$0 spend in the 90 days before cutoff, combined with 5 months of dormancy and multiple support tickets, pushed this customer's predicted probability from 49.87% to 98.83%.

---

### Case Study 2: High-Confidence Loyalist (True Negative)
* **Customer ID:** `500215` | **Tier:** Silver | **Store:** Mainstream | **City:** Nashville
* **Actual Outcome:** **Retained ($y = 0$)**
* **Model Output:** Predicted Churn Probability = **2.04%** (LR Probability = **0.00%**)
* **SHAP Verification:** $\text{Base Value } (0.4987) + \sum \text{SHAP } (-0.4783) = \mathbf{0.0204}$
* **Customer Profile:**
  - `TXNS_LAST_3M` = **27** transactions (~9 trips/month)
  - `SPEND_LAST_3M` = **$1,027.69**
  - `RECENCY_MONTHS_PRE_CUTOFF` = **0 months** (shopped in March 2024)
  - `TOTAL_SUPPORT_TICKETS_PRE_CUTOFF` = **0 tickets**
* **Top Features Driving Risk DOWN (Towards Retention):**
  - `SPEND_LAST_3M`: **$-0.1198$**
  - `TXNS_LAST_3M`: **$-0.1186$**
  - `RECENCY_MONTHS_PRE_CUTOFF`: **$-0.1063$**
* **Diagnostic Summary:** Heavy and continuous Q1 purchase momentum pushed churn probability down from 49.87% to 2.04%, correctly identifying an active customer.

---

### Case Study 3: Borderline / False Alarm (False Positive)
* **Customer ID:** `500414` | **Tier:** Silver | **Store:** Mainstream | **City:** Portland
* **Actual Outcome:** **Retained ($y = 0$)** *(Customer resumed shopping in Q2 2024)*
* **Model Output:** Predicted Churn Probability = **84.81%** (LR Probability = **68.27%**)
* **SHAP Verification:** $\text{Base Value } (0.4987) + \sum \text{SHAP } (+0.3494) = \mathbf{0.8481}$
* **Customer Profile:**
  - `TXNS_LAST_3M` = **1** transaction (only \$28.22 total spend in Q1)
  - `RECENCY_MONTHS_PRE_CUTOFF` = **2 months**
  - `TOTAL_SUPPORT_TICKETS_PRE_CUTOFF` = **0 tickets**
* **Top Features Driving Risk UP (Towards Churn):**
  - `SPEND_LAST_3M`: **$+0.0651$**
  - `RECENCY_MONTHS_PRE_CUTOFF`: **$+0.0643$**
  - `TXNS_LAST_3M`: **$+0.0577$**
* **Diagnostic Summary:** The customer experienced a temporary spending lull in early 2024 (1 small purchase of \$28 in 3 months). The model interpreted this drop as impending defection, pushing risk to 84.81%. This case study illustrates why false alarms occur on members with natural seasonal or travel lulls.

---

## 5. Artifact Verification & Execution Status

All Step 1 files have been updated, executed, and synchronized in the root workspace and [`phase_4_outputs/`](../../phase_4_outputs):

1. [phase4_step1_driver_analysis.py](../../phase4_step1_driver_analysis.py): Reproducible script that loads [`phase3_random_forest_pipeline.joblib`](../../phase3_random_forest_pipeline.joblib), verifies SHAP probability additivity, and exports tables and charts.
2. [phase4_step1_top_predictors.csv](../../phase4_step1_top_predictors.csv): Side-by-side ranking across RF Gini, SHAP, and LR standardized coefficients.
3. [phase4_step1_shap_summary.csv](../../phase4_step1_shap_summary.csv): Full table of 45 features ranked by validation cohort mean absolute SHAP value.
4. [phase4_step1_segment_risk_summary.csv](../../phase4_step1_segment_risk_summary.csv): Full reconciliation of all 602 validation customers by Tier and Store Price Tier.
5. [phase4_step1_city_risk_summary.csv](../../phase4_step1_city_risk_summary.csv): City-by-city validation breakdown with sample reliability warnings.
6. **Charts in [`phase_4_outputs/charts/`](../../phase_4_outputs/charts):**
   - `phase4_top_predictors_comparison.png`: RF Gini Impurity ranking.
   - `phase4_shap_summary.png`: Validation cohort SHAP attributions.
   - `phase4_segment_churn_profiles.png`: Actual vs. predicted churn rates by loyalty tier and store price tier.

---
*Phase 4 Step 1 quality review is complete. Execution has stopped before Step 2 (Feature Ablation Study).*
