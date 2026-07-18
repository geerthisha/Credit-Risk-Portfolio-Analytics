import pandas as pd
import numpy as np

# Load featured dataset matrix from Stage 2
df = pd.read_csv('Credit_Risk_Features.csv')

print("[INFO] Initializing Stage 3: Discretization & Weight of Evidence (WoE) / Information Value (IV) Analysis\n")

def calculate_woe_iv(dataset, feature, target):
    """
    Computes Weight of Evidence (WoE) and aggregated Information Value (IV)
    for a discretized credit scorecard variable.
    """
    # Group target matrices by specific interval bins
    grouped = dataset.groupby(feature)[target].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Bads']
    grouped['Goods'] = grouped['Total'] - grouped['Bads']
    
    # Apply standard risk engine adjustment to prevent undefined log zero fractions
    grouped['Bads'] = grouped['Bads'].replace(0, 0.5)
    grouped['Goods'] = grouped['Goods'].replace(0, 0.5)
    
    # Establish total portfolio volumes for baseline denominators
    total_goods = dataset[target].count() - dataset[target].sum()
    total_bads = dataset[target].sum()
    
    grouped['Dist_Goods'] = grouped['Goods'] / total_goods
    grouped['Dist_Bads'] = grouped['Bads'] / total_bads
    
    # Execute WoE alignment and part-IV transformation matrix
    grouped['WoE'] = np.log(grouped['Dist_Goods'] / grouped['Dist_Bads'])
    grouped['IV_Part'] = (grouped['Dist_Goods'] - grouped['Dist_Bads']) * grouped['WoE']
    
    iv = grouped['IV_Part'].sum()
    return grouped, iv


# --- CATEGORICAL INTERVAL VARIABLE DISCRETIZATION ---
# Map continuous data tracking profiles into risk bucket matrices
df_binned = pd.DataFrame()
df_binned['dlq_2yrs'] = df['dlq_2yrs']

# Profile A: Standard Age Cohorts
df_binned['age_bin'] = pd.cut(df['age'], bins=[18, 30, 40, 50, 60, 70, 100], include_lowest=True).astype(str)

# Profile B: Quantile Splitting for Income
df_binned['income_bin'] = pd.qcut(df['monthly_inc'], q=5, duplicates='drop').astype(str)

# Profile C: Quantile Splitting for Debt Load Metrics
df_binned['debt_bin'] = pd.qcut(df['debt_ratio'], q=5, duplicates='drop').astype(str)

# Profile D: Quantile Splitting for Line Utilization
df_binned['util_bin'] = pd.qcut(df['rev_util'], q=5, duplicates='drop').astype(str)

# Profile E: Custom Structured Bounds for Engineered Delinquency Metrics
df_binned['delinquency_bin'] = pd.cut(df['total_delinquency'], bins=[-1, 0, 1, 3, 5, 100], include_lowest=True).astype(str)


# --- STRATEGIC PREDICTIVE POWER PROFILING ---
features_to_test = ['age_bin', 'income_bin', 'debt_bin', 'util_bin', 'delinquency_bin']
iv_ranking = {}

for feat in features_to_test:
    woe_df, iv = calculate_woe_iv(df_binned, feat, 'dlq_2yrs')
    iv_ranking[feat] = iv
    
    # Print a diagnostic summary report using Age distribution as a validation check
    if feat == 'age_bin':
        print("[DEBUG] Diagnostic Variable Profile Validation (Feature: Age Cohorts)")
        print(woe_df[['Total', 'Bads', 'Goods', 'WoE']].round(4))
        print("-" * 65)

print("\n[INFO] Evaluated Information Value (IV) Predictive Rankings:")
print(f"{'Rank':<2} | {'Variable Name':<17} | {'Aggregated IV':<21} | {'Predictive Strength'}")
print("-" * 65)

sorted_iv = sorted(iv_ranking.items(), key=lambda x: x[1], reverse=True)
for rank, (feature, iv_val) in enumerate(sorted_iv, 1):
    strength = "Unviable"
    if iv_val >= 0.3: strength = "High Predictor"
    elif iv_val >= 0.1: strength = "Medium Predictor"
    elif iv_val >= 0.02: strength = "Weak Contextual"
    
    print(f"{rank:<2} | {feature:<17} | {iv_val:<21.4f} | {strength}")

print("-" * 65)
print("[SUCCESS] Variable statistical selection execution complete\n")