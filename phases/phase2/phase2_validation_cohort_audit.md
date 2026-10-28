# Phase 2: Validation Cohort Audit — FreshBasket Customer Churn Project

**Project:** Customer Churn Prediction for FreshBasket Retail Loyalty Program  
**Assessment:** Propel Internship Assessment  
  
**Date:** October 2026  
**Scope:** Phase 2 (Validation Strategy & Post-Cutoff Signup Audit)

---

## 1. Executive Summary: Cohort Definition & Temporal Integrity

The assessment brief mandates that **random train/test splitting must not be the primary evaluation method**, requiring instead a time-based or cohort-based validation split. Because the workbook contains only a single ground-truth churn outcome window (`2024-06-30`), we evaluate model generalizability using a **Signup Cohort-Based Validation Design**.

### 1.1 Temporal Boundaries of the Split
* **Prediction Cutoff Date:** **March 31, 2024 (23:59:59)**.
* **Training Cohort:** Loyalty members who enrolled on or before **June 30, 2023** (Mature customer base with $\ge 9$ months of pre-cutoff history).
* **Held-Out Validation Cohort:** Loyalty members who enrolled between **July 1, 2023 and March 31, 2024** (Newer customer cohorts entering their initial tenure window).
* **Post-Cutoff Signups (Excluded from Scoring):** Loyalty members who enrolled after **March 31, 2024** (April 1 – May 2, 2024).

```
========================================================================================
                      SIGNUP TIMELINE & COHORT PARTITIONING
========================================================================================
Feb 2021                       Jun 30, 2023                Mar 31, 2024      Jun 30, 2024
   |---------------------------------|---------------------------|----------------|
   [       TRAINING COHORT           ] [    VALIDATION COHORT    ] [ POST-CUTOFF  ]
   [      (Joined <= 2023-06-30)     ] [ (2023-07-01 - 2024-03-31) [ (Apr-May 2024) ]
   [         N = 1,947 (74.9%)       ] [     N = 602 (23.2%)     ] [ N = 51 (2.0%)]
                                                                 ^
                                                       Prediction Cutoff
```

---

## 2. Cohort Sizes, Churn Counts, and Class Distributions

The 2,600 loyalty members partition across the three groups as follows:

```
========================================================================================
                       VALIDATION COHORT DISTRIBUTION AUDIT
========================================================================================
Cohort Designation       Signup Date Range      Total Members    Retained (0)  Churned (1)  Churn Rate (%)
----------------------------------------------------------------------------------------
TRAIN                    2021-02-18 - 2023-06-30    1,947 (74.88%)   1,214         733         37.65%
VAL                      2023-07-01 - 2024-03-31      602 (23.15%)     517          85         14.12%
EXCLUDED_POST_CUTOFF     2024-04-01 - 2024-05-02       51  (1.96%)      51           0          0.00%
----------------------------------------------------------------------------------------
Total Dataset            2021-02-18 - 2024-05-02    2,600 (100.0%)   1,782         818         31.46%
========================================================================================
```

---

## 3. Audit of Customers Enrolled After the Prediction Cutoff

### 3.1 Identification and Label Analysis
* **Identified Count:** Exactly **51 loyalty members** in the dataset have `SIGNUP_DATE > 2024-03-31` (50 enrolled in April 2024, 1 enrolled on May 2, 2024).
* **Target Labels:** **100% of these 51 members are labelled `CHURNED = 0`**.
* **Why They Cannot Be Scored at Cutoff:**
  At the prediction cutoff of **March 31, 2024 (23:59:59)**, these 51 individuals were not yet loyalty members. In a real-world production deployment on March 31, 2024, a scoring pipeline could not possibly evaluate accounts that did not yet exist.
* **Methodological Risk of Inclusion:**
  If these 51 members were naively included in the validation evaluation set, they would artificially inject 51 trivial true negatives ($100\%$ non-churned accounts with 0 days tenure), artificially inflating validation specificity and accuracy.
* **Governing Decision:**
  These 51 customers are retained in `phase2_modeling_dataset.csv` for 100% population reconciliation, but are explicitly assigned `VALIDATION_COHORT = 'EXCLUDED_POST_CUTOFF'` and flagged with `IS_POST_CUTOFF_SIGNUP = 1`. They are **strictly excluded from the evaluation population** during Phase 3 model testing.

---

## 4. Evaluation of the Proposed Validation Strategy

### 4.1 Strengths of the June 30, 2023 Cohort Cutoff
1. **Realistic Temporal Generalization:** In retail, churn models are trained on historical customer histories to forecast attrition among newer incoming cohorts. Evaluating on members enrolled between July 2023 and March 2024 directly mirrors this operational deployment scenario.
2. **Standard Partition Ratios:** Among eligible members ($N = 2,549$), the split represents **76.4% Train (1,947 members)** and **23.6% Validation (602 members)**, matching standard machine learning partition guidelines (~75/25).
3. **Statistical Power for Metric Reliability:** The validation set contains **85 positive churn events**. Standard statistical power guidelines require $\ge 50$ positive events in test sets to ensure narrow confidence intervals around Precision-Recall curves and F1 scores.

---

### 4.2 Class Distribution Shift and Methodological Implications

The audit reveals a significant **class distribution shift** between training and validation cohorts:
* **Training Cohort Churn Rate:** **37.65%** (733 / 1,947)
* **Validation Cohort Churn Rate:** **14.12%** (85 / 602)

#### Why Does Churn Drop Among Newer Signups?
1. **Shorter Exposure Window:** Mature customers had up to 15 months to experience dissatisfaction, store operational friction, or life events causing them to drift away. Newer customers had only 1 to 8 months of membership before the April 2024 target quarter.
2. **Target Definition Filter:** Inactive signups who never made an initial purchase are classified as `CHURNED = 0` (unestablished shoppers). A higher proportion of late 2023/early 2024 signups fall into this category.

#### Methodological Protections for Phase 3:
1. **Failure of Default Thresholds ($0.50$):** A model trained on a $37.7\%$ base rate will naturally output higher probabilities than the $14.1\%$ test prevalence warrants. Applying a default decision threshold of $0.50$ will lead to depressed recall.
2. **Mandatory Threshold Optimization:** In Phase 3, the classification decision threshold must be calibrated or tuned on validation PR curves rather than defaulting to $0.50$.
3. **Primary Evaluation on PR-AUC:** **Precision-Recall AUC (PR-AUC)** evaluates the entire ranking spectrum without being tied to a fixed probability threshold, making it the most robust metric under cohort base-rate shift.
4. **Internal Cross-Validation Diagnostic:** To decouple model performance from cohort-specific base-rate differences, we will report **Stratified 5-Fold Cross-Validation on the Training Cohort** alongside the primary **Cohort Holdout Results**.
