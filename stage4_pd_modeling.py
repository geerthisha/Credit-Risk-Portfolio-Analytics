import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Load preprocessed feature space matrix from Stage 2
df = pd.read_csv('Credit_Risk_Features.csv')

print("[INFO] Initializing Stage 4: Probability of Default (PD) Statistical Modeling\n")

# --- CATEGORICAL INTERVAL DISCRETIZATION ALIGNMENT ---
# Replicate standard baseline risk bins across continuous parameters
df_binned = pd.DataFrame()
df_binned['dlq_2yrs'] = df['dlq_2yrs']
df_binned['age_bin'] = pd.cut(df['age'], bins=[18, 30, 40, 50, 60, 70, 100], include_lowest=True).astype(str)
df_binned['income_bin'] = pd.qcut(df['monthly_inc'], q=5, duplicates='drop').astype(str)
df_binned['debt_bin'] = pd.qcut(df['debt_ratio'], q=5, duplicates='drop').astype(str)
df_binned['util_bin'] = pd.qcut(df['rev_util'], q=5, duplicates='drop').astype(str)
df_binned['delinquency_bin'] = pd.cut(df['total_delinquency'], bins=[-1, 0, 1, 3, 5, 100], include_lowest=True).astype(str)

# --- ONE-HOT MATRIX TRANSFORMATION ---
# Convert categorical ranges to dummy indicators; drop first level to prevent collinear multi-variable inflation
X = pd.get_dummies(df_binned.drop(columns=['dlq_2yrs']), drop_first=True)
y = df_binned['dlq_2yrs']

# Explicitly cast boolean matrices to numeric integers for model compliance
X = X.astype(int)

# --- CROSS-VALIDATION PARTITIONING ---
# Split arrays into historical estimation subsets and out-of-sample testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"[DEBUG] Verification matrix: Subset partitioning array shapes:")
print(f"• Training Sub-array Shape : {X_train.shape}")
print(f"• Validation Test-array Shape: {X_test.shape}\n")

# --- HISTORICAL REGRESSION ESTIMATION ---
# Deploy regulatory-compliant logistic regression optimization engine
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Execute risk score evaluations on out-of-sample matrices
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --- PERFORMANCE CRITERIA METRICS ---
auc = roc_auc_score(y_test, y_prob)

print("[INFO] Out-of-Sample Performance Metric Analysis:")
print(classification_report(y_test, y_pred))
print(f"• Mapped Predictive Separation Power (AUC-ROC): {auc:.4f}")
print("-" * 65)

# --- RISK PARAMETER LOG-ODDS EXTRACTION ---
print("[DEBUG] Model Parameter Logs: Intercept and Optimized Log-Odds Coefficients")
print(f"Intercept: {model.intercept_[0]:.4f}\n")

coef_df = pd.DataFrame({
    'Feature_Bin_Threshold': X.columns,
    'Log_Odds_Coefficient': model.coef_[0]
}).sort_values(by='Log_Odds_Coefficient', ascending=False)

print(coef_df.to_string(index=False))

print("-" * 65)
print("[SUCCESS] Logistic Regression risk parameters optimized successfully\n")