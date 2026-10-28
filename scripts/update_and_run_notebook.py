import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell('''# FreshBasket Customer Churn Prediction
## Phase 3: Baseline Development, Model Training, and Cohort Validation

**Project:** FreshBasket Loyalty Customer Churn Prediction (Propel Assessment)  
**Objective:** Train and evaluate a Logistic Regression baseline and a Random Forest classifier using an out-of-cohort evaluation setup.  
**Source Dataset:** `phase2_modeling_dataset.csv` (2,600 rows × 33 columns).  
**Cohorts:**
* **Training Cohort:** Members enrolled on or before June 30, 2023 ($N = 1,947$, Churn Rate = $37.65\%$).
* **Validation Cohort:** Members enrolled July 1, 2023 through March 31, 2024 ($N = 602$, Churn Rate = $14.12\%$).
* **Excluded:** Post-cutoff signups after March 31, 2024 ($N = 51$, not available at cutoff).

> **Generalization & Cohort Shift Note:**  
> The validation cohort exhibits a significantly lower churn rate ($14.12\%$) than the training cohort ($37.65\%$) because newer members had shorter pre-cutoff observation periods (1–8 months vs $\ge 9$ months). Both models use balanced class weighting on training data, meaning raw predicted probabilities will tend to over-estimate churn risk on newer cohorts. Neither model is considered production-ready without probability calibration and threshold alignment.'''),

    nbf.v4.new_markdown_cell('''### 1. Ingestion & Data Preparation
We load the master Phase 2 modeling dataset and partition into training and validation cohorts, strictly excluding post-cutoff enrollments and administrative/leakage columns.'''),

    nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)

# Load dataset
df = pd.read_csv('phase2_modeling_dataset.csv')

# Partition cohorts
train_df = df[df['VALIDATION_COHORT'] == 'TRAIN'].copy()
val_df = df[df['VALIDATION_COHORT'] == 'VAL'].copy()
excluded_df = df[df['VALIDATION_COHORT'] == 'EXCLUDED_POST_CUTOFF'].copy()

print(f"Training Cohort:   {len(train_df):,} rows | Churned: {train_df['CHURNED'].sum()} ({train_df['CHURNED'].mean()*100:.2f}%)")
print(f"Validation Cohort: {len(val_df):,} rows | Churned: {val_df['CHURNED'].sum()} ({val_df['CHURNED'].mean()*100:.2f}%)")
print(f"Excluded Signups:  {len(excluded_df):,} rows (Signed up after 2024-03-31)")

# Separate features from administrative columns
admin_cols = ['CUSTOMER_ID', 'SIGNUP_DATE', 'CHURNED', 'VALIDATION_COHORT', 'IS_POST_CUTOFF_SIGNUP']
feature_cols = [c for c in df.columns if c not in admin_cols]

cat_cols = ['GENDER', 'CITY', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER']
num_cols = [c for c in feature_cols if c not in cat_cols]

X_train, y_train = train_df[feature_cols], train_df['CHURNED']
X_val, y_val = val_df[feature_cols], val_df['CHURNED']

print(f"\\nInput Features: {len(feature_cols)} total ({len(num_cols)} numerical, {len(cat_cols)} categorical)")'''),

    nbf.v4.new_markdown_cell('''### 2. Preprocessing Pipelines
We construct scikit-learn `Pipeline` and `ColumnTransformer` pipelines. Preprocessors are fit strictly on the training cohort to eliminate data leakage.'''),

    nbf.v4.new_code_cell('''# Preprocessing for Logistic Regression (with scaling)
num_transformer_lr = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for Random Forest (without scaling)
num_transformer_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor_lr = ColumnTransformer([
    ('num', num_transformer_lr, num_cols),
    ('cat', cat_transformer, cat_cols)
])

preprocessor_rf = ColumnTransformer([
    ('num', num_transformer_rf, num_cols),
    ('cat', cat_transformer, cat_cols)
])'''),

    nbf.v4.new_markdown_cell('''### 3. Model Training
We fit two distinct model architectures on the training cohort:
1. **Logistic Regression Baseline** (Linear model with balanced class weighting).
2. **Random Forest Classifier** (Non-linear tree ensemble with balanced class weighting).'''),

    nbf.v4.new_code_cell('''# 1. Logistic Regression Baseline
lr_model = Pipeline([
    ('pre', preprocessor_lr),
    ('clf', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
])
lr_model.fit(X_train, y_train)

# 2. Random Forest Classifier
rf_model = Pipeline([
    ('pre', preprocessor_rf),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])
rf_model.fit(X_train, y_train)

print("Both models fitted successfully on training cohort.")'''),

    nbf.v4.new_markdown_cell('''### 4. Out-of-Cohort Evaluation & Trade-off Analysis
We generate probability estimates and class predictions on the held-out validation cohort ($N = 602$).

**Performance Trade-off Summary:**
* **Logistic Regression:** Achieves higher Precision ($89.55\%$), F1-Score ($0.7895$), and PR-AUC ($0.8808$), with only 7 False Positives. It is optimal if retention interventions are expensive or high-touch.
* **Random Forest:** Achieves higher Recall ($82.35\%$ vs $70.59\%$), identifying 70 of 85 churners ($FN = 15$). It is preferred if missing churners is far costlier than sending low-cost retention communications.
* The preferred model depends on the business ratio of intervention costs to missed defection costs.'''),

    nbf.v4.new_code_cell('''# Predictions and probabilities
y_val_pred_lr = lr_model.predict(X_val)
y_val_proba_lr = lr_model.predict_proba(X_val)[:, 1]

y_val_pred_rf = rf_model.predict(X_val)
y_val_proba_rf = rf_model.predict_proba(X_val)[:, 1]

def evaluate_predictions(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1_Score': round(f1, 4),
        'PR_AUC': round(pr_auc, 4),
        'ROC_AUC': round(roc_auc, 4),
        'TN': int(cm[0,0]), 'FP': int(cm[0,1]),
        'FN': int(cm[1,0]), 'TP': int(cm[1,1])
    }

metrics = [
    evaluate_predictions('Logistic Regression (Baseline)', y_val, y_val_pred_lr, y_val_proba_lr),
    evaluate_predictions('Random Forest Classifier (Primary)', y_val, y_val_pred_rf, y_val_proba_rf)
]

metrics_df = pd.DataFrame(metrics)
print("=== Validation Cohort Performance Table ===")
display(metrics_df)'''),

    nbf.v4.new_markdown_cell('''### 5. Feature Drivers & Interpretability
* **Logistic Regression:** Because numeric features were standardized, coefficients represent the change in log-odds associated with a **one-standard-deviation increase** in the predictor, holding other standardized features constant ($\text{Odds Ratio} = \exp(\beta)$). These represent observational predictive associations, not causal effects.
* **Random Forest Gini Importance:** Measures the mean decrease in node impurity (MDI) across splits in the ensemble. It captures predictive contribution within this specific model structure, not causal impact.'''),

    nbf.v4.new_code_cell('''# Extract feature names after One-Hot Encoding
ohe = lr_model.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
all_feature_names = num_cols + cat_feature_names

# Logistic Regression Coefficients
lr_coefs = lr_model.named_steps['clf'].coef_[0]
df_lr_imp = pd.DataFrame({
    'Feature': all_feature_names,
    'LR_Coefficient': lr_coefs,
    'Odds_Ratio': np.exp(lr_coefs)
}).sort_values(by='LR_Coefficient', key=abs, ascending=False)

# Random Forest Gini Importances
rf_importances = rf_model.named_steps['clf'].feature_importances_
df_rf_imp = pd.DataFrame({
    'Feature': all_feature_names,
    'RF_Gini_Importance': rf_importances
}).sort_values(by='RF_Gini_Importance', ascending=False)

print("Top 10 Random Forest Predictors (Gini Impurity Importance):")
display(df_rf_imp.head(10))

print("\\nTop 10 Logistic Regression Coefficients (Standardized Scale):")
display(df_lr_imp.head(10))'''),

    nbf.v4.new_markdown_cell('''### 6. Subgroup Performance Comparison
Performance across loyalty tiers and store price positioning. Subgroups with small sample sizes (e.g., Platinum with $N = 63$, Unknown Store Price Tier with $N = 23$) exhibit wide sampling variability and should not be used for definitive conclusions.'''),

    nbf.v4.new_code_cell('''val_analysis = val_df[['CUSTOMER_ID', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER', 'CHURNED']].copy()
val_analysis['LR_PRED'] = y_val_pred_lr
val_analysis['RF_PRED'] = y_val_pred_rf

print("Performance by Membership Tier (Validation Cohort):")
for tier, group in val_analysis.groupby('MEMBERSHIP_TIER'):
    n = len(group)
    churn_n = group['CHURNED'].sum()
    tp_rf = ((group['CHURNED'] == 1) & (group['RF_PRED'] == 1)).sum()
    fp_rf = ((group['CHURNED'] == 0) & (group['RF_PRED'] == 1)).sum()
    rec_rf = tp_rf / churn_n if churn_n > 0 else 0
    prec_rf = tp_rf / (tp_rf + fp_rf) if (tp_rf + fp_rf) > 0 else 0
    print(f"Tier: {tier:8s} | N={n:3d} | Churn={churn_n:2d} ({churn_n/n*100:4.1f}%) | RF Recall={rec_rf:4.2f}, Precision={prec_rf:4.2f}")'''),

    nbf.v4.new_markdown_cell('''### 7. Exporting Predictions and Artifacts
We export validation predictions with customer IDs, predicted labels, and probabilities.'''),

    nbf.v4.new_code_cell('''val_export = val_df[['CUSTOMER_ID', 'SIGNUP_DATE', 'MEMBERSHIP_TIER', 'HOME_STORE_PRICE_TIER', 'CHURNED']].copy()
val_export['LR_CHURN_PROBABILITY'] = y_val_proba_lr.round(4)
val_export['LR_PRED_LABEL'] = y_val_pred_lr
val_export['RF_CHURN_PROBABILITY'] = y_val_proba_rf.round(4)
val_export['RF_PRED_LABEL'] = y_val_pred_rf
val_export['PRIMARY_MODEL_NAME'] = 'RandomForest_Balanced'
val_export['PRIMARY_MODEL_PROBABILITY'] = val_export['RF_CHURN_PROBABILITY']
val_export['PRIMARY_MODEL_PRED_LABEL'] = val_export['RF_PRED_LABEL']

val_export.to_csv('phase3_validation_predictions.csv', index=False)
print("Exported 'phase3_validation_predictions.csv' successfully.")''')
]

nb.cells = cells
with open('phase3_model_development_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Created phase3_model_development_notebook.ipynb successfully.")

# Execute notebook to pre-render all outputs
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
print("Executing notebook end-to-end...")
ep.preprocess(nb, {'metadata': {'path': '.'}})

with open('phase3_model_development_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook executed and all outputs saved successfully!")
