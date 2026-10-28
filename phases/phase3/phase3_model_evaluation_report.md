# Phase 3 Model Evaluation Report: Baseline Development, Model Training, and Cohort Validation (Revised)
**FreshBasket Loyalty Customer Churn Prediction Project (Propel Internship Assessment)**  
*Date: October 2026*

---

## Executive Summary

Phase 3 transitions the FreshBasket Customer Churn Prediction project from feature engineering into empirical model training and out-of-cohort evaluation. Following the approved Phase 2 modeling dataset (`phase2_modeling_dataset.csv`, 2,600 rows × 33 columns), we developed:
1. **A Logistic Regression Baseline** (linear benchmark with balanced class weighting and standardized numerical inputs).
2. **A Random Forest Classifier** (non-linear decision tree ensemble with balanced class weighting, max depth of 6, min leaf samples of 10, and 200 estimators).

Both models were fitted **strictly on the Training Cohort** (customers enrolled on or before June 30, 2023; $N = 1,947$) and evaluated on the **Holdout Validation Cohort** (customers enrolled July 1, 2023 through March 31, 2024; $N = 602$). All 51 post-cutoff enrollments (signed up after March 31, 2024) were strictly excluded from model inputs and evaluation.

### Verified Validation Performance at Default Decision Threshold (0.50)
* **Logistic Regression Baseline:** Accuracy = **94.68%**, Precision = **89.55%**, Recall = **70.59%**, F1-Score = **0.7895**, PR-AUC = **0.8808**, ROC-AUC = **0.9644** ($TN = 510, FP = 7, FN = 25, TP = 60$).
* **Random Forest Classifier:** Accuracy = **91.69%**, Precision = **66.67%**, Recall = **82.35%**, F1-Score = **0.7368**, PR-AUC = **0.8499**, ROC-AUC = **0.9446** ($TN = 482, FP = 35, FN = 15, TP = 70$).

### Key Model-Selection Takeaway: A Nuanced Business Trade-off
Rather than one model being universally superior, the evaluation demonstrates a clear trade-off:
* **Logistic Regression** is the preferred model when retention interventions are **costly or capacity-constrained** (e.g., personal outreach, high-value physical gifts) because its high precision ($89.55\%$) and high PR-AUC ($0.8808$) minimize wasted spend (only 7 false alarms).
* **Random Forest** is the preferred model at the default 0.50 threshold when retention interventions are **low-cost and automated** (e.g., digital coupon delivery, email re-engagement) and the cost of missing an at-risk customer dominates, as it captures **82.35% of churners** (70 out of 85), missing only 15.

> **Production-Readiness Notice:**  
> Neither model is claimed to be production-ready. The validation cohort exhibits substantial class shift ($14.12\%$ churn rate vs. $37.65\%$ in training) due to shorter observation windows for newer members, causing probability over-estimation in Random Forest (mean predicted ~30% vs 14.12% actual, whereas Logistic Regression is close to calibrated at ~13.9%) that requires post-hoc probability calibration, dynamic threshold tuning, and operational pipeline validation before automated deployment.

---

## 1. Cohort & Data Verification Against Phase 2

All row counts, customer IDs, and class distributions were reconciled against the finalized Phase 2 dataset:

| Cohort | Definition | Total Count | Retained ($y=0$) | Churned ($y=1$) | Churn Rate | Status in Phase 3 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Cohort** | Signed up on or before 2023-06-30 ($\ge 9$ months pre-cutoff) | **1,947** | 1,214 | 733 | **37.65%** | Fitted preprocessor & model |
| **Validation Cohort** | Signed up 2023-07-01 to 2024-03-31 ($< 9$ months pre-cutoff) | **602** | 517 | 85 | **14.12%** | Out-of-cohort evaluation |
| **Excluded Signups** | Signed up after 2024-03-31 (Post-cutoff enrollments) | **51** | 51 | 0 | **0.00%** | Excluded completely |
| **Total Master Dataset** | Full cleaned customer population | **2,600** | 1,782 | 818 | **31.46%** | Reconciled with Phase 1 & 2 |

### Administrative and Leakage Features Excluded from Model Inputs
To guarantee zero target leakage and prevent identity memorization, the following 5 columns were strictly removed prior to preprocessing and training:
1. `CUSTOMER_ID` (Unique identifier)
2. `SIGNUP_DATE` (Cohort assignment delimiter)
3. `CHURNED` (Target variable)
4. `VALIDATION_COHORT` (Cohort assignment label)
5. `IS_POST_CUTOFF_SIGNUP` (Administrative post-cutoff indicator)

This leaves exactly **28 predictor features** (24 numerical, 4 categorical) available for modeling.

---

## 2. Cohort Churn Rate Disparity & Generalization Limitations

A critical finding from the cohort setup is the large divergence in baseline churn prevalence:
* **Training Cohort Churn Rate:** **37.65%** (733 churned out of 1,947)
* **Validation Cohort Churn Rate:** **14.12%** (85 churned out of 602)

### Root Cause of the Difference
This disparity does not reflect an operational improvement in FreshBasket's retention, but rather a structural feature of customer tenure:
1. **Exposure Duration:** Customers in the training cohort enrolled on or before June 30, 2023, providing 9 to 15 months of pre-cutoff history. Over a longer observation period, more customers naturally reach the end of their shopping lifecycle or encounter a dissatisfying experience.
2. **Newer Signups:** Validation cohort customers enrolled between July 1, 2023, and March 31, 2024. Those joining in early 2024 had only 1 to 3 months of history before the cutoff, meaning fewer had completed a normal lifecycle or entered dormancy.

### Methodological & Generalization Consequences
* **Impact of Balanced Class Weights:** Both models were trained using `class_weight='balanced'`, which re-weights training samples to an effective 50:50 ratio. When applied to the validation cohort where the true prevalence is only 14.12%, Random Forest systematically over-estimates churn likelihood (mean predicted probability 30.36% vs. 14.12% actual), leading to elevated false alarms and lower precision, while Logistic Regression remains close to calibrated on average (mean predicted probability 13.86%).
* **Generalization Limits:** Performance estimates obtained on this validation cohort reflect performance on newer customers with limited tenure. The models cannot be assumed to perform identically across cohorts with different tenure distributions without tenure-stratified modeling or post-hoc probability calibration (e.g., Platt scaling or isotonic regression).

---

## 3. Preprocessing Architecture & Leakage Prevention

All preprocessing transformations were encapsulated within `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer` constructs, fitted **strictly on the 1,947 training rows**:

```
                              ┌──────────────────────────────────────────────┐
                              │  Training Cohort Only (N = 1,947)           │
                              └──────────────────────┬───────────────────────┘
                                                     │
                                       ColumnTransformer.fit()
                                                     │
                     ┌───────────────────────────────┴───────────────────────────────┐
                     ▼                                                               ▼
        ┌─────────────────────────┐                                     ┌─────────────────────────┐
        │   Numerical Features    │                                     │  Categorical Features   │
        │      (24 columns)       │                                     │       (4 columns)       │
        └────────────┬────────────┘                                     └────────────┬────────────┘
                     │                                                               │
        SimpleImputer(median)                                           SimpleImputer(Unknown)
                     │                                                               │
        StandardScaler() [LR only]                                      OneHotEncoder(handle_unknown='ignore')
                     │                                                               │
                     └───────────────────────────────┬───────────────────────────────┘
                                                     ▼
                                       Concatenated Feature Vector
                                        (45 transformed features)
```

1. **Numerical Imputation:** Missing values were imputed with the training-set median.
2. **Standardization:** Numerical features were standardized (mean = 0, standard deviation = 1) for Logistic Regression. Standardization was omitted in Random Forest to preserve tree scale-invariance.
3. **Categorical Handling:** Missing categories were assigned `'Unknown'`. Nominal features (`GENDER`, `CITY`, `MEMBERSHIP_TIER`, `HOME_STORE_PRICE_TIER`) were one-hot encoded with `handle_unknown='ignore'` to safely handle unseen levels.

---

## 4. Comprehensive Model Evaluation & Metrics Table

Both models were evaluated on the exact same 602 validation customers using the default decision threshold ($P \ge 0.50$):

| Evaluation Metric | Logistic Regression (Baseline) | Random Forest Classifier | Relative Comparison / Analytical Note |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **0.9468** (94.7%) | 0.9169 (91.7%) | LR higher due to fewer false alarms on majority class |
| **Precision** | **0.8955** (89.6%) | 0.6667 (66.7%) | **LR leads by 22.9 percentage points** (7 vs. 35 FP) |
| **Recall (Sensitivity)** | 0.7059 (70.6%) | **0.8235** (82.4%) | **RF captures +10 additional churners** (70 vs. 60 TP) |
| **F1-Score** | **0.7895** | 0.7368 | LR leads in harmonic mean due to higher precision |
| **PR-AUC (Avg Precision)**| **0.8808** | 0.8499 | Evaluates precision-recall across all threshold cutoffs |
| **ROC-AUC** | **0.9644** | 0.9446 | Both models demonstrate strong global rank-ordering |
| **True Negatives ($TN$)** | **510** | 482 | Correctly classified retained customers |
| **False Positives ($FP$)**| **7** | 35 | Retained customers incorrectly flagged as churners |
| **False Negatives ($FN$)**| 25 | **15** | Missed churners who leave without detection |
| **True Positives ($TP$)** | 60 | **70** | Successfully detected churners |

---

## 5. Confusion Matrix Analysis & Trade-off Economics

### Confusion Matrices on Validation Cohort ($N = 602$, Actual Churners = 85, Retained = 517)

```
        Logistic Regression (Baseline)                Random Forest Classifier
                 Predicted                                         Predicted
             Retained   Churned                                Retained   Churned
Actual   0  [  510   |    7   ]                     Actual 0  [  482   |   35   ]
         1  [   25   |   60   ]                            1  [   15   |   70   ]
```

### Practical Business Meaning of Classification Errors

1. **False Positives ($FP$ — "False Alarms"):**
   - **Definition:** The model predicts a customer will churn, but they would have remained active without intervention.
   - **Operational Impact:** A retained customer receives an unneeded retention offer (e.g., promotional discount or targeted email).
   - **Cost to Business:** Cost of the promotional incentive, unnecessary marketing operational load, and potential habituation of loyal customers to discounts.
   - **Counts:** LR = 7 ($1.4\%$ of retained cohort), RF = 35 ($6.8\%$ of retained cohort).

2. **False Negatives ($FN$ — "Silent Defections"):**
   - **Definition:** The model predicts a customer is safe, but they quietly cease shopping in Q2 2024.
   - **Operational Impact:** No retention outreach occurs. The customer leaves for a competitor unaddressed.
   - **Cost to Business:** Loss of future customer lifetime spend and retail margin.
   - **Counts:** LR = 25 missed churners ($29.4\%$ missed), RF = 15 missed churners ($17.6\%$ missed).

### Hypothetical Scenario for Decision Threshold Alignment (Illustrative Only)

> [!NOTE]
> **Methodological Disclaimer on Retention Economics:**  
> The financial evaluation of churn models depends heavily on operational parameters that are currently **unverified** by FreshBasket:
> 1. *Offer Cost:* The actual cost per targeted customer depends on campaign type (e.g., \$0.05 for email vs. \$10 for a mailer/voucher).
> 2. *Rescue Rate:* In practice, not all targeted churners can be saved; typical retail retention campaign lift ranges from 10% to 30%.
> 3. *Customer Margin:* Gross margin per customer varies widely by loyalty tier and spending category.

To illustrate how business parameters determine model choice, consider two contrasting operational scenarios:

* **Scenario A: High-Touch / High-Cost Intervention (e.g., \$25 Gift Basket or Dedicated Account Outreach)**
  - Cost per False Alarm: **\$25**
  - Net Value of 1 Saved Customer: **\$100** (assuming a 25% rescue rate on \$400 annual margin)
  - Under these parameters, Random Forest's 28 extra False Positives incur **\$700 in wasted outreach**, whereas its 10 extra True Positives generate **\$250 in net value**.
  - **Result:** **Logistic Regression is the superior choice** because avoiding false alarms preserves budget.

* **Scenario B: Low-Cost / Automated Digital Intervention (e.g., \$1.00 Personalized In-App Promotion / Email)**
  - Cost per False Alarm: **\$1.00**
  - Net Value of 1 Saved Customer: **\$50** (assuming a 10% rescue rate on \$500 annual margin)
  - Here, Random Forest's 28 extra False Positives cost only **\$28**, while its 10 extra True Positives generate **\$500 in value**.
  - **Result:** **Random Forest is the superior choice** because capturing more churners heavily outweighs the low cost of digital false alarms.

**Conclusion:** Neither model is universally optimal. Model selection and threshold tuning must directly reflect the specific cost structure of FreshBasket's planned retention campaign.

---

## 6. Threshold Decision & Sensitivity Analysis

The official operating threshold for all primary deliverables remains **0.50** for both models. This threshold was selected as the standard unadjusted classification boundary **without post-hoc tuning on the validation cohort**, because the validation cohort contains only 85 actual churners ($N = 602$), making threshold search on validation data prone to over-fitting and optimistic bias.

### 6.1 Cohort Prevalence Disparity and Calibration Implications
A fundamental characteristic of this dataset is the significant cohort churn prevalence divergence:
* **Training Cohort:** **37.65%** churn prevalence (733 churners / 1,947 members; members had 10 to 15 months of pre-cutoff exposure).
* **Validation Cohort:** **14.12%** churn prevalence (85 churners / 602 members; newer signups with 1 to 9 months of exposure).

Evaluating model calibration on the 602 validation members reveals a vital distinction between the architectures:
* **Logistic Regression:** Produces a mean predicted probability of **0.1386** (~13.9%), which is close to calibrated on average compared with the actual observed churn rate of **14.12%**.
* **Random Forest:** Produces a mean predicted probability of **0.3036** (~30.4%), substantially over-estimating the 14.12% actual churn prevalence due to balanced class weighting applied to the higher-prevalence training cohort. This systematic over-estimation explains why Random Forest flags considerably more members as at-risk (105 predicted churners vs. 67 for Logistic Regression) and consequently exhibits lower precision (66.67% vs. 89.55%).

Despite Logistic Regression's overall mean alignment, predicted probabilities across both models should be treated primarily as **relative risk scores for customer ranking** rather than fully calibrated deployment probabilities until a formal calibration review (e.g., Platt scaling, isotonic regression) and an operational threshold review aligned with intervention economics are conducted prior to live deployment.

### 6.2 Supplementary Sensitivity Analysis: Default 0.50 vs. Training-Only 5-Fold CV Thresholds
To evaluate threshold sensitivity without touching validation data, 5-fold cross-validation was run strictly on the **training cohort only** to identify the out-of-fold F1-maximizing threshold for each architecture ($\tau^*_{\text{LR}} = 0.60$, $\tau^*_{\text{RF}} = 0.53$). Applying these training-derived thresholds to the holdout validation cohort reveals the empirical trade-off:

| Model | Threshold Strategy | Threshold ($\tau$) | Accuracy | Precision | Recall | F1-Score | True Neg ($TN$) | False Pos ($FP$) | False Neg ($FN$) | True Pos ($TP$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **Default (Official)** | **0.50** | **0.9468** | 0.8955 | **0.7059** | **0.7895** | 510 | 7 | 25 | 60 |
| **Logistic Regression** | Training-CV Optimized | 0.60 | 0.9452 | **0.9194** | 0.6706 | 0.7755 | **512** | **5** | 28 | 57 |
| **Random Forest** | **Default (Official)** | **0.50** | **0.9169** | 0.6667 | **0.8235** | **0.7368** | 482 | 35 | 15 | 70 |
| **Random Forest** | Training-CV Optimized | 0.53 | **0.9169** | **0.6768** | 0.7882 | 0.7283 | **485** | **32** | 18 | 67 |

*(Saved as `phases/phase3/output/phase3_threshold_comparison.csv`. This analysis is strictly supplementary; all official assessment numbers remain anchored at 0.50).*

---

## 7. Bootstrap Confidence Intervals (Validation Cohort, N = 602)

To evaluate the sampling uncertainty of validation metrics without retraining, a non-parametric bootstrap analysis (1,000 resamples, fixed seed = 42) was performed on the existing validation cohort predictions (`phase3_validation_predictions.csv`).

Because the validation cohort contains only **85 actual churners**, sampling variability is substantial and confidence intervals are relatively wide.

| Model | Metric | Observed Value | Bootstrap Mean | 95% Percentile Interval [2.5%, 97.5%] |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | **PR-AUC** | 0.8808 | 0.8803 | [0.8256, 0.9287] |
| **Logistic Regression** | **ROC-AUC** | 0.9644 | 0.9641 | [0.9466, 0.9799] |
| **Logistic Regression** | **Precision (0.50)** | 0.8955 | 0.8977 | [0.8182, 0.9678] |
| **Logistic Regression** | **Recall (0.50)** | 0.7059 | 0.7053 | [0.6111, 0.7977] |
| **Logistic Regression** | **F1-Score (0.50)** | 0.7895 | 0.7888 | [0.7172, 0.8571] |
| **Random Forest** | **PR-AUC** | 0.8499 | 0.8492 | [0.7884, 0.9058] |
| **Random Forest** | **ROC-AUC** | 0.9446 | 0.9440 | [0.9155, 0.9698] |
| **Random Forest** | **Precision (0.50)** | 0.6667 | 0.6677 | [0.5833, 0.7568] |
| **Random Forest** | **Recall (0.50)** | 0.8235 | 0.8218 | [0.7317, 0.9036] |
| **Random Forest** | **F1-Score (0.50)** | 0.7368 | 0.7358 | [0.6666, 0.8043] |

### 7.1 Paired Bootstrap Test on PR-AUC Difference (LR minus RF)
To assess whether the observed PR-AUC advantage of Logistic Regression over Random Forest ($+0.0310$) is statistically significant, a paired resample difference ($\Delta_{\text{PR-AUC}} = \text{PR-AUC}_{\text{LR}} - \text{PR-AUC}_{\text{RF}}$) was computed across all 1,000 replicates:
* **Observed Difference:** $+0.0310$
* **Bootstrap Mean Difference:** $+0.0311$
* **95% Percentile Confidence Interval:** **[-0.0022, 0.0677]**
* **Interval Contains Zero:** **Yes**

**Statistical & Practical Takeaway:**  
Because the 95% paired confidence interval spans zero (lower bound $-0.0022$), there is **no statistically significant difference** in PR-AUC between Logistic Regression and Random Forest on this validation cohort. Neither model should be claimed as clearly superior. The established strategic recommendation is confirmed: **Random Forest is preferred for high-recall, low-cost interventions** (capturing 82.35% of defections), while **Logistic Regression is preferred when precision is paramount** to minimize wasted spend on costly interventions.

*(Saved as `phases/phase3/output/phase3_bootstrap_confidence_intervals.csv`).*

---

## 8. Model Interpretability & Feature Predictors

### 6.1 Logistic Regression Standardized Coefficients (Observational Associations)

Because numeric features were standardized prior to fitting ($\mu = 0, \sigma = 1$), each Logistic Regression coefficient $\beta_j$ represents the change in log-odds associated with a **one-standard-deviation increase** in feature $j$, holding all other standardized features constant:
$$\text{Odds Ratio} = \exp(\beta_j)$$

> [!IMPORTANT]
> **Non-Causal Interpretation:**  
> These coefficients reflect observational statistical associations within the training data, **not causal effects**. For example, a positive coefficient on support tickets indicates that customers with higher ticket counts were more likely to churn in the historical data; it does *not* imply that artificially suppressing customer support would reduce churn.

```
                  Top Standardized Logistic Regression Coefficients
  ┌─────────────────────────────────────┬──────────────┬────────────┬─────────────────────────────┐
  │ Feature Name                        │ LR Coeff (β) │ Odds Ratio │ Training Scale (Std Dev σ)  │
  ├─────────────────────────────────────┼──────────────┼────────────┼─────────────────────────────┤
  │ TXNS_LAST_3M                        │   -3.0280    │   0.0484   │ σ = 8.14 transactions       │
  │ SPEND_LAST_3M                       │   -2.0986    │   0.1226   │ σ = $294.70                 │
  │ PRE_CUTOFF_OBSERVED_MONTHS          │   -1.9024    │   0.1492   │ σ = 2.45 months             │
  │ AVG_MONTHLY_TXNS_PRE_CUTOFF         │   +0.8643    │   2.3732   │ σ = 2.21 transactions/mo    │
  │ AVG_MONTHLY_EMAILS_OPENED           │   -0.6959    │   0.4986   │ σ = 0.61 emails/mo          │
  │ TOTAL_SUPPORT_TICKETS_PRE_CUTOFF    │   +0.6281    │   1.8740   │ σ = 1.37 tickets            │
  │ RECENCY_MONTHS_PRE_CUTOFF           │   +0.6130    │   1.8460   │ σ = 20.84 months (w/ 99 fill│
  │ TOTAL_APP_SESSIONS_PRE_CUTOFF       │   -0.5161    │   0.5968   │ σ = 16.98 sessions          │
  └─────────────────────────────────────┴──────────────┴────────────┴─────────────────────────────┘
```

#### Detailed Predictor Explanations
1. **Recent Shopping Velocity (`TXNS_LAST_3M` & `SPEND_LAST_3M`):**
   - Holding other standardized features constant, a one-standard-deviation increase in transactions during the last 3 months ($+8.14$ txns) is associated with an estimated **$95.2\%$ decrease** in the odds of churn ($\text{Odds Ratio} = 0.048$).
   - Recent shopping activity is the strongest empirical discriminator of customer retention.
2. **Support Interaction (`TOTAL_SUPPORT_TICKETS_PRE_CUTOFF`):**
   - A one-standard-deviation increase in support tickets ($+1.37$ tickets) is associated with an estimated **$87.4\%$ increase** in the odds of churn ($\text{Odds Ratio} = 1.874$).
   - Frequent customer service contacts statistically correlate with defection, likely reflecting unresolved product quality or fulfillment issues.
3. **Digital Touchpoints (`AVG_MONTHLY_EMAILS_OPENED` & `TOTAL_APP_SESSIONS`):**
   - A one-standard-deviation increase in monthly email opens ($+0.61$ emails/mo) is associated with an estimated **$50.1\%$ decrease** in the odds of churn ($\text{Odds Ratio} = 0.499$).
   - Engagement with FreshBasket's digital channels correlates with brand retention.

---

### 6.2 Random Forest Feature Importance (Predictive MDI, Not Causation)

Random Forest feature importance is computed using Gini impurity reduction (Mean Decrease in Impurity / MDI):

```
                     Top 10 Random Forest Predictors (MDI)
  ┌─────────────────────────────────────┬──────────────────┬────────────────────────┐
  │ Feature Name                        │ Gini Importance  │ Predictive Role        │
  ├─────────────────────────────────────┼──────────────────┼────────────────────────┤
  │ TXNS_LAST_3M                        │      17.41%      │ Strongest split factor │
  │ RECENCY_MONTHS_PRE_CUTOFF           │      16.86%      │ Primary recency signal │
  │ SPEND_LAST_3M                       │      16.49%      │ Monetary momentum      │
  │ AVG_MONTHLY_TXNS_PRE_CUTOFF         │       6.68%      │ Baseline visit rate    │
  │ ACTIVE_MONTHS_PRE_CUTOFF            │       6.51%      │ Customer tenure depth  │
  │ AVG_MONTHLY_SPEND_PRE_CUTOFF        │       5.70%      │ Baseline monetary rate │
  │ TOTAL_TXNS_PRE_CUTOFF               │       4.99%      │ Lifetime volume        │
  │ AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF │       4.79%      │ App engagement rate    │
  │ PRE_CUTOFF_OBSERVED_MONTHS          │       4.01%      │ Exposure window        │
  │ TOTAL_SPEND_PRE_CUTOFF              │       3.65%      │ Lifetime monetary      │
  └─────────────────────────────────────┴──────────────────┴────────────────────────┘
```

> [!NOTE]
> **Methodological Properties of Gini Importance:**  
> 1. **Predictive Contribution Only:** Gini importance quantifies how frequently and effectively a feature was selected to partition nodes across the 200 trees. It measures predictive relevance within this ensemble structure, **not a real-world causal mechanism**.
> 2. **Split Cardinality Bias:** Impurity-based importance is known to favor continuous features with many candidate split thresholds (such as spend and transaction totals) over categorical features (such as gender or tier).
> 3. **Correlated Feature Sharing:** Features that share information (e.g., `TXNS_LAST_3M` and `SPEND_LAST_3M`) split importance across trees, which may depress their individual ranks relative to an orthogonal model.

---

## 9. Segment-Level Validation Performance & Sample Size Limitations

To assess performance consistency across customer subgroups, we evaluated both models across loyalty tiers and store price tiers.

> [!WARNING]
> **Sample Size Caution for Subgroups:**  
> Several customer subgroups have very small sample sizes in the validation cohort. For example, **Platinum members comprise only 63 customers (14 churners)**, and **Unknown store tier comprises only 23 customers (0 churners)**. With small counts, a single misclassification causes large percentage swings (e.g., 1 customer in Platinum represents $7.1\%$ of recall). Consequently, subgroup metrics should be viewed as exploratory rather than statistically definitive.

### 7.1 Membership Tier Breakdown (Validation Cohort, $N = 602$)

| Segment | Total ($N$) | Churned ($N$) | Churn Rate | Model | Accuracy | Precision | Recall | F1-Score | $TP$ | $FP$ | $FN$ | $TN$ |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Silver** | 348 | 45 | 12.9% | **Logistic Regression** | 94.5% | 86.1% | 68.9% | 0.765 | 31 | 5 | 14 | 298 |
| | | | | **Random Forest** | 91.7% | 64.3% | 80.0% | 0.713 | 36 | 20 | 9 | 283 |
| **Gold** | 191 | 26 | 13.6% | **Logistic Regression** | 96.9% | 95.5% | 80.8% | 0.875 | 21 | 1 | 5 | 164 |
| | | | | **Random Forest** | 92.7% | 66.7% | 92.3% | 0.774 | 24 | 12 | 2 | 153 |
| **Platinum** | 63 | 14 | 22.2% | **Logistic Regression** | 88.9% | 88.9% | 57.1% | 0.696 | 8 | 1 | 6 | 48 |
| *(Small Sample)* | | | | **Random Forest** | 88.9% | 76.9% | 71.4% | 0.741 | 10 | 3 | 4 | 46 |

**Analytical Observations:**
* On Silver and Gold members, Random Forest achieves higher recall (80.0% and 92.3%) at the expense of lower precision (64.3% and 66.7%), while Logistic Regression maintains precision above 86%.
* On Platinum members ($N = 63$), both models achieve 88.9% accuracy. Random Forest identifies 10 of 14 churners (71.4% recall) with 3 false positives, while Logistic Regression identifies 8 of 14 churners (57.1% recall) with 1 false positive. Due to the small sample of 14 churners, confidence intervals are wide.

### 7.2 Home Store Price Tier Breakdown (Validation Cohort, $N = 602$)

| Store Price Tier | Total ($N$) | Churned ($N$) | Churn Rate | Model | Accuracy | Precision | Recall | F1-Score | $TP$ | $FP$ | $FN$ | $TN$ |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mainstream** | 281 | 41 | 14.6% | **Logistic Regression** | 95.4% | 91.2% | 75.6% | 0.827 | 31 | 3 | 10 | 237 |
| | | | | **Random Forest** | 91.5% | 64.9% | 90.2% | 0.755 | 37 | 20 | 4 | 220 |
| **Value** | 198 | 27 | 13.6% | **Logistic Regression** | 93.4% | 88.9% | 59.3% | 0.711 | 16 | 2 | 11 | 169 |
| | | | | **Random Forest** | 89.9% | 62.1% | 66.7% | 0.643 | 18 | 11 | 9 | 160 |
| **Upscale** | 100 | 17 | 17.0% | **Logistic Regression** | 94.0% | 86.7% | 76.5% | 0.812 | 13 | 2 | 4 | 81 |
| | | | | **Random Forest** | 94.0% | 78.9% | 88.2% | 0.833 | 15 | 4 | 2 | 79 |
| **Unknown** | 23 | 0 | 0.0% | **Both Models** | 100.0% | N/A | N/A | N/A | 0 | 0 | 0 | 23 |
| *(Small Sample)* | | | | | | | | | | | | |

**Analytical Observations:**
* On Mainstream and Upscale customers, Random Forest attains 90.2% and 88.2% recall, capturing a large proportion of defections.
* On Value store customers ($N = 198$), both models exhibit lower recall (59.3% for LR, 66.7% for RF), suggesting that budget-oriented shoppers have more volatile or deal-sensitive purchase patterns that are moderately less predictable from general transaction recency alone.
* The Unknown store tier ($N = 23$) contains zero churners; precision and recall are mathematically undefined.

---

## 10. Summary of Revisions & Deliverables

All deliverables have been verified, re-executed, and synchronized in the root workspace and `phases/phase3/output/`:

1. **`phase3_model_development_notebook.ipynb`**: Re-executed end-to-end with revised markdown annotations clarifying non-causal interpretations, the cohort churn-rate gap, and the trade-off between LR and RF.
2. **`phase3_validation_predictions.csv`**: Contains all 602 validation customers with customer IDs, actual churn labels, and model probabilities. Verified to match reported metrics.
3. **`phase3_model_metrics.csv`**: Reconciled metrics table with exact confusion matrix values.
4. **`phase3_threshold_comparison.csv`**: Supplementary sensitivity analysis comparing default 0.50 threshold against training-only 5-fold CV F1-maximizing threshold.
5. **`phase3_bootstrap_confidence_intervals.csv`**: 1,000-resample non-parametric bootstrap confidence intervals and paired PR-AUC difference evaluation.
6. **`phase3_logistic_regression_coefficients.csv`**: Full standardized coefficient and odds-ratio table for 45 model features.
7. **`phase3_random_forest_feature_importances.csv`**: Gini impurity-based importance table.
8. **`phase3_model_evaluation_report.md`**: This revised evaluation report.

---
*Phase 3 is complete and fully reconciled. Phase 4 will not begin until these deliverables have been reviewed.*
