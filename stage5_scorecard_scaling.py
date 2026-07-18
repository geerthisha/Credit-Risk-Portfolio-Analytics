import pandas as pd
import numpy as np

# Load preprocessed feature space matrix from Stage 2
df = pd.read_csv('Credit_Risk_Features.csv')

print("[INFO] Initializing Stage 5: Log-Odds Scorecard Points Scaling Pipeline\n")

# --- CATEGORICAL BIN ALIGNMENT REPLICATION ---
df_binned = pd.DataFrame()
df_binned['dlq_2yrs'] = df['dlq_2yrs']
df_binned['age_bin'] = pd.cut(df['age'], bins=[18, 30, 40, 50, 60, 70, 100], include_lowest=True).astype(str)
df_binned['income_bin'] = pd.qcut(df['monthly_inc'], q=5, duplicates='drop').astype(str)
df_binned['debt_bin'] = pd.qcut(df['debt_ratio'], q=5, duplicates='drop').astype(str)
df_binned['util_bin'] = pd.qcut(df['rev_util'], q=5, duplicates='drop').astype(str)
df_binned['delinquency_bin'] = pd.cut(df['total_delinquency'], bins=[-1, 0, 1, 3, 5, 100], include_lowest=True).astype(str)

X = pd.get_dummies(df_binned.drop(columns=['dlq_2yrs']), drop_first=True).astype(int)

# Map optimized risk parameters extracted from the Stage 4 regression fit
intercept = -1.4883
coefficients = {
    'delinquency_bin_(5.0, 100.0]': 3.057203,
    'delinquency_bin_(3.0, 5.0]': 2.397699,
    'delinquency_bin_nan': 2.263484,
    'delinquency_bin_(1.0, 3.0]': 1.752596,
    'util_bin_(0.983, 1.448]': 1.727783,
    'util_bin_(0.65, 0.983]': 1.464679,
    'delinquency_bin_(0.0, 1.0]': 0.953132,
    'util_bin_(0.263, 0.65]': 0.883470,
    'debt_bin_(0.604, 367.88]': 0.620219,
    'util_bin_(0.0509, 0.263]': 0.247477,
    'debt_bin_(0.394, 0.604]': 0.192241,
    'income_bin_(2800.0, 4200.0]': 0.092257,
    'debt_bin_(0.255, 0.394]': 0.049223,
    'income_bin_(5833.0, 8400.0]': -0.035565,
    'income_bin_(4200.0, 5833.0]': -0.035682,
    'debt_bin_(0.119, 0.255]': -0.046489,
    'age_bin_(30.0, 40.0]': -0.066351,
    'income_bin_(8400.0, 250000.0]': -0.105131,
    'age_bin_(40.0, 50.0]': -0.134486,
    'age_bin_(50.0, 60.0]': -0.312141,
    'age_bin_(70.0, 100.0]': -0.657473,
    'age_bin_(60.0, 70.0]': -0.659657
}

# --- SCALING ALGORITHM MATRIX ---
# Configure traditional baseline parameters: 600 points mapping to 1:1 odds ratio | PDO = 20
factor = 20 / np.log(2)
offset = 600.0

print(f"[DEBUG] Core Scaling Scaling Parameters:")
print(f"• Mathematical Scaling Factor: {factor:.4f}")
print(f"• Operational Score Offset    : {offset:.4f}\n")

# Distribute proportional baseline weights across active predictive paths
num_features = 5
base_points = (offset / num_features) - (factor * intercept / num_features)

scorecard_rows = []
for col in X.columns:
    coef = coefficients