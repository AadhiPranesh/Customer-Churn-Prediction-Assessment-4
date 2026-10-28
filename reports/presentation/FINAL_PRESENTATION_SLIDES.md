# FreshBasket Customer Churn Prediction — Slide Content Only
## Tailwyndz Technologies Propel Internship Assessment 4

**Candidate:** Aadhi Pranesh SS  
**Department:** Computer Science and Engineering  
**Institution:** Kongu Engineering College  
**Google Slides Link:** https://docs.google.com/presentation/d/1pI8quuO_O1MC8b6-e9jm4_fHYE3un5Dh/edit?usp=sharing&ouid=105329998739422544837&rtpof=true&sd=true



## Slide 1: Title & Executive Overview

FRESHBASKET RETAIL LOYALTY PROGRAM
Customer Churn Prediction & Proactive Retention Engine

Project: Customer Churn Prediction
Technology: Python, Pandas, Scikit-learn, SHAP
Presented by: Aadhi Pranesh SS
Department: Computer Science and Engineering
College: Kongu Engineering College

---

## Slide 2: Business Problem & Project Objectives

BUSINESS PROBLEM & PROJECT OBJECTIVES

Business Problem
• FreshBasket may lose customers who stop purchasing.
• Current retention offers may reach customers who do not need them.
• High-risk customers may leave without targeted intervention.

Project Objectives
• Predict customers at risk of churn using machine learning.
• Identify the key factors associated with customer churn.
• Compare Logistic Regression and Random Forest.
• Recommend targeted customer retention strategies.

---

## Slide 3: Dataset & Data Quality

DATASET & DATA QUALITY

Dataset Overview
• Total customers: 2,600
• Primary data tables: 3 (Customer Profile, Monthly Activity, Churn Label)
• Activity period: January 2023 – June 2024 (18 months)

Data Cleaning
• Removed 8 duplicate profile rows.
• Removed 5 duplicate churn-label rows.
• Removed 60 activity rows: 58 exact duplicates and 2 conflicting rows.
• Handled negative and zero-spend records.
• Treated missing demographic values.
• Preserved 6 customers with no activity records.

---

## Slide 4: Churn Definition & Data Leakage Prevention

CHURN DEFINITION & DATA LEAKAGE PREVENTION

Churn Definition
• Cut-off date: March 31, 2024
• Prediction period: April 1 – June 30, 2024
• Churn means zero purchases during the prediction period, despite previous purchase history.

Temporal Validation
• Training cohort: 1,947 customers; 733 churners (37.65% churn)
• Validation cohort: 602 customers; 85 churners (14.12% churn)
• Excluded: 51 customers who signed up after the cut-off date

Leakage Prevention
• Use only information available before April 1, 2024, as model predictors.
• Keep the future churn period separate for evaluation.

---

## Slide 5: Feature Engineering & Model Development

FEATURE ENGINEERING & MODEL DEVELOPMENT

Feature Engineering
• 28 predictor features from customer data
• Customer profile and demographics (8 features)
• Purchase behaviour and recent shopping activity (11 features)
• Customer engagement and support (6 features)
• Data quality and activity indicators (3 features)

Machine Learning Models
• Logistic Regression: Baseline classification model
• Random Forest: Ensemble model using 200 decision trees
• Missing values handled using imputation
• Categorical variables converted using One-Hot Encoding
• Standard scaling applied to Logistic Regression

---

## Slide 6: Model Evaluation & Comparison

MODEL EVALUATION & COMPARISON

Metric                      Logistic Regression    Random Forest
Accuracy                    94.68%                 91.69%
Precision                   89.55% (60 / 67)       66.67% (70 / 105)
Recall                      70.59% (60 / 85)       82.35% (70 / 85)
F1-score                    78.95%                 73.68%
PR-AUC                      0.8808                 0.8499
Validation results – 602 customers, decision threshold 0.50

Key Findings
• Logistic Regression: Higher precision and fewer false alarms (7 vs 35).
• Random Forest: Higher recall; detects 70 of 85 churners (misses only 15).
• The difference in PR-AUC is not statistically significant based on the paired confidence interval.

---

## Slide 7: Churn Drivers & SHAP Explainability

CHURN DRIVERS & SHAP EXPLAINABILITY

Top Random Forest Predictors
• Recent transactions – TXNS_LAST_3M (0.1741)
• Purchase recency – RECENCY_MONTHS_PRE_CUTOFF (0.1686)
• Recent spending – SPEND_LAST_3M (0.1649)

SHAP Analysis
• SHAP explains how features influence individual model predictions.
• Recent purchase activity and spending are important churn signals.
• These relationships show associations, not proven causes.

Example Customer: 500014
• Predicted churn risk: 77.67%
• Top drivers: recent spending and recent transactions

---

## Slide 8: Feature Ablation & Customer Segmentation

FEATURE ABLATION & CUSTOMER SEGMENTATION

Feature group                 PR-AUC
Demographics                  0.1410
Demographics + History        0.1791
Purchase Behaviour            0.8290
Full Feature Set              0.8499

Feature Ablation Study
• Comparison using Random Forest validation results.

Customer Segmentation
• Silver churn rate: 12.93% (45 / 348)
• Gold churn rate: 13.61% (26 / 191)
• Platinum churn rate: 22.22% (14 / 63 customers; small sample)
• Signup cohorts show different observed churn rates.

---

## Slide 9: Model Limitations & Challenges

MODEL LIMITATIONS & CHALLENGES

1. Limited Validation Data
• Only 602 customers in the validation cohort.
• Only 85 actual churners.

2. Probability Calibration
• Logistic Regression mean predicted risk: 13.86%.
• Random Forest mean predicted risk: 30.36%.
• Formal probability calibration is still required.

3. Data Limitations
• Predictions depend on the available customer records.
• The analysis identifies associations, not proven causes of churn.

4. Business Impact
• Retention results are simulated, not measured in a real campaign.
• Actual effectiveness must be tested before deployment.

---

## Slide 10: Conclusion & Retention Recommendations

CONCLUSION & RETENTION RECOMMENDATIONS

Recommended Retention Strategies
• Support follow-up: Contact customers with recorded support issues.
• Reactivation offers: Target customers with long purchase gaps.
• Digital engagement: Reach customers with low digital activity.

Key Project Findings
• Logistic Regression achieved 89.55% precision.
• Random Forest achieved 82.35% recall.
• Purchase behaviour was an important predictor of customer churn.

Business Value
• Identify customers at risk of leaving.
• Support targeted retention campaigns.
• Help reduce wasted marketing spending.
