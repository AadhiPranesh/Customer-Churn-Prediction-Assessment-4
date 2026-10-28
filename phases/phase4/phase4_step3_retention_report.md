# Phase 4 — Step 3: Retention Intervention Simulation Report (Revised)
**FreshBasket Loyalty Customer Churn Prediction Project (Propel Internship Assessment)**  
*Date: October 2026*

---

## Executive Summary

Phase 4 Step 3 evaluates how the validated churn risk scores could guide **targeted, scenario-based retention playbooks**. Rather than treating all at-risk customers as a homogeneous pool, we simulate three tailored, pre-cutoff intervention strategies targeted at specific behavioral disengagement symptoms.

> [!IMPORTANT]
> **Methodological Disclaimer — Scenario Analysis, Not a Causal Experiment:**  
> This simulation is an **assumption-based scenario analysis**, not an empirical causal experiment or validated A/B test. 
> 1. The model identifies observational risk associations; it cannot guarantee that targeted customers will respond to an offer.
> 2. Projected customer savings across the **10%, 20%, and 30% sensitivity range** are **hypothetical illustrations**, not predictions validated by historical campaign outcomes.
> 3. No unverified revenue, profit, or return on investment (ROI) claims are made, as FreshBasket has not established campaign delivery costs or customer margin baselines.

### Key Step 3 Takeaways
* **Target Population:** Based on the primary Phase 3 Random Forest model ($P \ge 0.50$), **105 validation customers** are identified as at-risk. This group captures **70 of the 85 actual validation churners** ($82.35\%$ recall) alongside 35 false alarms ($66.67\%$ precision).
* **Targeting Precision by Behavioral Symptom:**
  * **Scenario 1 (Support Follow-up):** Targeting customers with prior service friction (`TOTAL_SUPPORT_TICKETS > 0` or `COMPLAINT == 1`) captures **54 customers**, of whom **45 actually churned** (**83.3% precision**).
  * **Scenario 2 (Reactivation Offer):** Targeting customers with purchase dormancy ($\text{recency of at least two months}$, i.e., `RECENCY_MONTHS_PRE_CUTOFF >= 2`) captures **55 customers**, of whom **52 actually churned** (**94.5% precision**).
  * **Scenario 3 (Digital Re-engagement):** Targeting digitally disengaged members (zero monthly app sessions or zero monthly email opens) captures **51 customers**, of whom **35 actually churned** (**68.6% precision**).
* **Deduplicated Program Scale:** Across the three targeted playbooks, **91 unique customers** are addressed (86.7% of the at-risk cohort), capturing **67 actual churners** ($78.8\%$ of all validation churners).
* **Hypothetical Churn Reduction Impact (Illustrative Sensitivity Range):**
  * At a conservative **10% hypothetical efficacy**, the combined program saves an estimated **~6.7 customers**.
  * At a moderate **20% hypothetical efficacy**, the program saves an estimated **~13.4 customers**.
  * At an optimistic **30% hypothetical efficacy**, the program saves an estimated **~20.1 customers** (preventing almost one-quarter of all validation defections).

---

## 1. Scenario Definitions & Pre-Cutoff Targeting Rules

Each scenario addresses a specific, interpretable defection signal identified in earlier phases. All targeting rules rely **strictly on information available before the March 31, 2024 cutoff date**:

```
                               Targeting Architecture
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Primary Model At-Risk Population (N = 105, 70 Churned, 35 Retained)   │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
  ┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
  │ Scenario 1: Support    │  │ Scenario 2: Recency    │  │ Scenario 3: Digital    │
  │ • SUPPORT_TICKETS > 0  │  │ • RECENCY >= 2 months  │  │ • Monthly App == 0     │
  │   OR COMPLAINT == 1    │  │   (recency >= 2 mos)   │  │   OR Monthly Email == 0│
  │ • N = 54 (45 churners) │  │ • N = 55 (52 churners) │  │ • N = 51 (35 churners) │
  └────────────────────────┘  └────────────────────────┘  └────────────────────────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Combined Deduplicated Program: 91 Unique Customers (67 Actual Churners)│
  └────────────────────────────────────────────────────────────────────────┘
```

### Clarification on `RECENCY_MONTHS_PRE_CUTOFF` Calculation:
* In Phase 2 feature engineering, `RECENCY_MONTHS_PRE_CUTOFF` was constructed as:
  $$\text{RECENCY\_MONTHS\_PRE\_CUTOFF} = (2024 - \text{MONTH.year}) \times 12 + (3 - \text{MONTH.month})$$
  measuring the elapsed count of calendar months between March 2024 and the customer's latest pre-cutoff month with purchase transactions ($> 0$).
* Customers who shopped in March 2024 have recency $= 0$; February 2024 $= 1$; January 2024 $= 2$; December 2023 $= 3$; etc.
* Because activity is recorded at the monthly boundary, this variable measures **calendar-month recency** rather than an exact elapsed day count. Consequently, Scenario 2 is strictly defined as **"recency of at least two months"** (`RECENCY_MONTHS_PRE_CUTOFF >= 2`), avoiding misleading claims of exact day intervals.

### Playbook Details:
1. **Scenario 1: Customer Support Follow-Up & Friction Resolution**
   * *Targeting Rule:* At-risk customers with recorded customer support activity or formal complaints prior to cutoff (`TOTAL_SUPPORT_TICKETS_PRE_CUTOFF > 0` OR `COMPLAINT_RECORDED_PRE_CUTOFF == 1`).
   * *Operational Intervention:* High-touch service recovery outreach by customer experience staff to resolve unaddressed service breakdowns, offer product replacements, or issue goodwill loyalty points.
2. **Scenario 2: Reactivation Offer for Inactivity (Recency $\ge 2$ Months)**
   * *Targeting Rule:* At-risk customers with purchase recency of at least two calendar months prior to cutoff (`RECENCY_MONTHS_PRE_CUTOFF >= 2`, i.e., no recorded purchases in February or March 2024).
   * *Operational Intervention:* High-value promotional voucher (e.g., \$15 off a \$50 grocery basket) designed to trigger a store re-visit before the customer forms a habit with a competing supermarket.
3. **Scenario 3: Digital Re-engagement for Disengaged Members**
   * *Targeting Rule:* At-risk customers with zero digital activity in at least one primary channel (`AVG_MONTHLY_APP_SESSIONS_PRE_CUTOFF == 0` OR `AVG_MONTHLY_EMAILS_OPENED_PRE_CUTOFF == 0`).
   * *Operational Intervention:* Multi-channel digital re-boarding campaign featuring personalized digital circulars, app download bonuses, and mobile-only product discounts.
4. **Combined Program (Deduplicated Unique Customers):**
   * *Targeting Rule:* Any at-risk customer qualifying under Scenario 1, Scenario 2, OR Scenario 3. Customers qualifying for multiple scenarios are assigned a tiered, coordinated touchpoint rather than receiving duplicate uncoordinated marketing messages.

---

## 2. Retention Scenario Simulation Results Table

The following table details the targeting scale, empirical precision, and **hypothetical illustrations** of saved customers across the sensitivity range:

| Scenario | Targeting Rule | Targeted ($N$) | % of At-Risk | Actual Churners | False Alarms ($FP$) | Precision (%) | % of All Val Churners | Hypothetical Saved (10% Efficacy)* | Hypothetical Saved (20% Efficacy)* | Hypothetical Saved (30% Efficacy)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1: Support Follow-up** | Tickets $>0$ or Complaint $=1$ | 54 | 51.4% | 45 | 9 | 83.3% | 52.9% (45/85) | 4.5 | 9.0 | 13.5 |
| **S2: Reactivation Offer**| Recency $\ge 2$ months | 55 | 52.4% | 52 | 3 | **94.5%** | 61.2% (52/85) | 5.2 | 10.4 | 15.6 |
| **S3: Digital Re-engagement**| Monthly App $=0$ or Email $=0$| 51 | 48.6% | 35 | 16 | 68.6% | 41.2% (35/85) | 3.5 | 7.0 | 10.5 |
| **Combined (Deduplicated)**| **S1 OR S2 OR S3 (Unique)** | **91** | **86.7%** | **67** | **24** | **73.6%** | **78.8% (67/85)**| **6.7** | **13.4** | **20.1** |

*\*Note: Figures in the efficacy columns are assumption-based hypothetical illustrations, not predictions validated by historical intervention outcomes.*

---

## 3. Overlap Analysis & Complete Cohort Reconciliation

Because customers frequently exhibit multiple concurrent symptoms of defection, individual scenario counts cannot be summed directly. The table below provides a complete, mutually exclusive decomposition of all 105 at-risk members, including those not captured by scenarios 1–3:

| Customer Targeting Subgroup | Description | Total Count ($N$) | Actual Churners | False Alarms ($FP$) | Group Churn Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Scenario 1 Only** | Support friction only (recent shoppers, digitally active) | 12 | 7 | 5 | 58.3% |
| **Scenario 2 Only** | Recency $\ge 2$ mos only (no tickets, digitally active) | 13 | 10 | 3 | 76.9% |
| **Scenario 3 Only** | Digital disengagement only (recent shoppers, no tickets) | 14 | 2 | 12 | 14.3% |
| **S1 and S2 Only** | Service friction + Recency $\ge 2$ mos (digitally active) | 15 | 15 | 0 | **100.0%** |
| **S1 and S3 Only** | Service friction + Digital disengagement (recent shoppers) | 10 | 6 | 4 | 60.0% |
| **S2 and S3 Only** | Recency $\ge 2$ mos + Digital disengagement (no tickets) | 10 | 10 | 0 | **100.0%** |
| **All Three Scenarios** | Service friction + Recency $\ge 2$ mos + Digital disengagement| 17 | 17 | 0 | **100.0%** |
| **Subtotal: Unique Targeted**| **Qualifies for $\ge 1$ Scenario (S1, S2, or S3)** | **91** | **67** | **24** | **73.6%** |
| **Not Targeted by S1–S3** | At-risk due to declining spend velocity alone | 14 | 3 | 11 | 21.4% |
| **Total At-Risk Population** | **Model Predicted Churners ($P \ge 0.50$)** | **105** | **70** | **35** | **66.7%** |

### Complete Cohort Reconciliation Checks:
* **Sum of All Subgroups:** $12 + 13 + 14 + 15 + 10 + 10 + 17 + 14 = \mathbf{105}$ (Exact Match).
* **Sum of Actual Churners:** $7 + 10 + 2 + 15 + 6 + 10 + 17 + 3 = \mathbf{70}$ (Exact Match to True Positives).
* **Sum of False Alarms:** $5 + 3 + 12 + 0 + 4 + 0 + 0 + 11 = \mathbf{35}$ (Exact Match to False Positives).
* **Total Validation Churn Coverage:** The 67 actual churners targeted by the combined program represent **$78.8\%$** of all 85 actual churners in the validation cohort.

---

## 4. Sensitivity Analysis & Hypothetical Efficacy Ranges

To provide leadership with realistic operational expectations, we illustrate potential customer savings across three hypothetical efficacy benchmarks:

1. **Conservative Illustration (10% Hypothetical Efficacy):**
   * *Assumption:* 1 in 10 targeted churners is persuaded to resume regular shopping.
   * *Hypothetical Result:* Saves **~4.5 churners** in S1, **~5.2 churners** in S2, or **~6.7 unique churners** across the combined program.
   * *Business Application:* Benchmarks low-touch digital communications (e.g., automated email notifications or app pushes).
2. **Moderate Illustration (20% Hypothetical Efficacy):**
   * *Assumption:* 1 in 5 targeted churners is saved through an appealing financial incentive or personalized phone outreach.
   * *Hypothetical Result:* Saves **~9.0 churners** in S1, **~10.4 churners** in S2, or **~13.4 unique churners** across the combined program.
   * *Business Application:* Benchmarks targeted retail reactivation vouchers (\$10–\$15 off next purchase).
3. **Optimistic Illustration (30% Hypothetical Efficacy):**
   * *Assumption:* High-touch concierge outreach or VIP resolution successfully resolves root friction for nearly 1 in 3 members.
   * *Hypothetical Result:* Saves **~13.5 churners** in S1, **~15.6 churners** in S2, or **~20.1 unique churners** across the combined program.
   * *Business Application:* Achievable in high-tier segments (e.g., dedicated Gold/Platinum service recovery).

---

## 5. Methodological Limitations & Financial Prudence

> [!WARNING]
> **Why Revenue and Profit Projections Are Omitted:**  
> Claiming net profit or ROI without verified financial parameters is a serious methodological flaw. We deliberately avoid asserting dollar gains because:
> 1. *Redemption and Fulfillment Costs:* The cost of promotional vouchers (\$10 vs. \$15), call center labor, and postage is not documented in the assessment brief.
> 2. *Cannibalization / Margin Erosion:* Vouchers sent to the 24 false alarms (retained customers who would have shopped anyway) erode grocery gross margin.
> 3. *Baseline Margin Variability:* Average annual spend varies widely across customer tiers (\$500 to \$2,500+). Without margin data by category, financial projections remain speculative.
> 
> *The simulation serves as an assumption-based illustration to guide campaign capacity planning and A/B test design, not a guaranteed financial prediction.*

---

## 6. Deliverables Created & Verification Status

All Step 3 code, tabular results, customer rosters, and charts have been generated and verified in the workspace root and [`phase_4_outputs/`](../../phase_4_outputs):

1. [phase4_step3_retention_simulation.py](../../phase4_step3_retention_simulation.py): Reproducible simulation script running all scenario rules, overlap logic, and sensitivity calculations.
2. [phase4_step3_retention_scenario_results.csv](../../phase4_step3_retention_scenario_results.csv): Tabular metrics summary for S1, S2, S3, and Combined programs.
3. [phase4_step3_scenario_overlap.csv](../../phase4_step3_scenario_overlap.csv): Full mutually exclusive overlap decomposition of all 105 at-risk customers.
4. [phase4_step3_customer_targeting_roster.csv](../../phase4_step3_customer_targeting_roster.csv): Customer-level targeting flags for all 105 at-risk members with customer IDs and probabilities.
5. [phase_4_outputs/charts/phase4_retention_simulation.png](../../phase_4_outputs/charts/phase4_retention_simulation.png): Two-panel visualization comparing targeting precision and saved customer sensitivity ranges.
6. [phase4_step3_retention_report.md](../../phase4_step3_retention_report.md): This revised comprehensive report.

---
*Phase 4 — Step 3 quality review is complete. Execution has stopped before the final report and presentation.*
