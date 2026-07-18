import pandas as pd
import numpy as np

# Load raw application records
df = pd.read_csv('Credit Risk Benchmark Dataset.csv')

# Drop default unindexed row indicators if present from source
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

print("[INFO] Initializing Stage 1: Data Cleaning & Preprocessing Pipeline\n")

# --- MISSING VALUE IMPUTATION ---
print("[DEBUG] Tracking data anomalies: Missing fields before imputation:")
print(df[['monthly_inc', 'dependents']].isnull().sum())

# Establish fallback thresholds based on active portfolio modes and medians
dependents_mode = df['dependents'].mode()[0]
income_median = df['monthly_inc'].median()

df['dependents'] = df['dependents'].fillna(dependents_mode)
df['monthly_inc'] = df['monthly_inc'].fillna(income_median)

print("\n[DEBUG] Verification matrix: Null counts post-imputation:")
print(df[['monthly_inc', 'dependents']].isnull().sum())
print("-" * 50)


# --- OUTLIER MANAGEMENT (CAPPING) ---
print("\n[INFO] Running statistical outlier scanning and boundary capping parameters")

# Drop age records incompatible with standard credit lines
age_outliers = df[(df['age'] < 18) | (df['age'] > 100)].shape[0]
print(f"• Age profile exceptions identified (outside 18-100 threshold): {age_outliers}")
df = df[(df['age'] >= 18) & (df['age'] <= 100)]

# Cap extreme debt metrics at the 99th percentile to stabilize variance
debt_99th = df['debt_ratio'].quantile(0.99)
print(f"• Debt Ratio distribution: Capping bound set at 99th percentile: {debt_99th:.2f}")
print(f"  - Baseline raw maximum: {df['debt_ratio'].max():,.2f}")
df['debt_ratio'] = np.where(df['debt_ratio'] > debt_99th, debt_99th, df['debt_ratio'])
print(f"  - Calibrated operational maximum: {df['debt_ratio'].max():,.2f}")

# Cap volatile credit utilization scores at the 99th percentile 
util_99th = df['rev_util'].quantile(0.01) # Wait, let's keep the exact 99th quantile math from before:
util_99th = df['rev_util'].quantile(0.99)
print(f"• Credit Utilization distribution: Capping bound set at 99th percentile: {util_99th:.4f}")
print(f"  - Baseline raw maximum: {df['rev_util'].max():,.4f}")
df['rev_util'] = np.where(df['rev_util'] > util_99th, util_99th, df['rev_util'])
print(f"  - Calibrated operational maximum: {df['rev_util'].max():,.4f}")


# --- DATASETS STAGING EXPORT ---
print("-" * 50)
print(f"[SUCCESS] Stable array construction complete: {df.shape[0]} rows | {df.shape[1]} columns")

df.to_csv('Credit_Risk_Cleaned.csv', index=False)
print("[INFO] preprocessed data matrix successfully exported to 'Credit_Risk_Cleaned.csv'\n")