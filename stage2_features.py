import pandas as pd

# Load processed data matrix from Stage 1
df = pd.read_csv('Credit_Risk_Cleaned.csv')

print("[INFO] Initializing Stage 2: Behavioral Feature Engineering Pipeline\n")

# Map tracking columns for historical payment disruptions
delinquency_cols = ['late_30_59', 'late_60_89', 'late_90']

if all(col in df.columns for col in delinquency_cols):
    # Construct a composite index weighting severity across risk horizons:
    # 30-59 days past due (Weight = 1)
    # 60-89 days past due (Weight = 2)
    # 90+ days past due   (Weight = 3)
    df['total_delinquency'] = (df['late_30_59'] * 1) + (df['late_60_89'] * 2) + (df['late_90'] * 3)
    print("[SUCCESS] Composite feature vector 'total_delinquency' successfully generated.")
    
    # Review feature frequency and spread dynamics
    print("\n[DEBUG] Distribution summary metrics for 'total_delinquency':")
    print(df['total_delinquency'].describe())
    
    # Map empirical default probabilities grouped by feature scores
    print("\n[DEBUG] Observed default rate percentage tracked by risk index:")
    risk_profile = df.groupby('total_delinquency')['dlq_2yrs'].mean() * 100
    print(risk_profile.head(10).apply(lambda x: f"{x:.2f}%"))
    
else:
    print("[ERROR] Missing required operational columns in data matrix source.")
    print(f"Available features index: {df.columns.tolist()}")

print("-" * 50)
print(f"[SUCCESS] Array construction modified: {df.shape[0]} rows | {df.shape[1]} columns")

# Export expanded feature space matrix
df.to_csv('Credit_Risk_Features.csv', index=False)
print("[INFO] Features matrix successfully written to 'Credit_Risk_Features.csv'\n")