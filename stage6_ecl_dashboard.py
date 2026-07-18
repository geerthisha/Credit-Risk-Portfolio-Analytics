import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure standardized visualization properties
sns.set_theme(style='whitegrid')

# Load the scored portfolio data matrix from Stage 5
df = pd.read_csv('Portfolio_Scores.csv')

print("[INFO] Initializing Stage 6: Segmentation & Forward-Looking IFRS-9 ECL Analytics\n")

# --- ACCOUNT SEGMENTATION AND REPRICING TIERS ---
def get_credit_tier(score):
    """Maps continuous scaled points to discrete institutional risk classes."""
    if score < 530:
        return '1. Deep Subprime (<530)'
    elif score < 580:
        return '2. Subprime (530-580)'
    elif score < 620:
        return '3. Near Prime (580-620)'
    elif score < 660:
        return '4. Prime (620-660)'
    else:
        return '5. Super Prime (>=660)'

df['credit_tier'] = df['credit_score'].apply(get_credit_tier)

# --- EMPIRICAL LEVEL GROUPING ANALYSIS ---
tier_analysis = df.groupby('credit_tier').agg(
    total_borrowers=('dlq_2yrs', 'count'),
    actual_defaults=('dlq_2yrs', 'sum'),
).reset_index()

# Extract point-in-time observed Empirical Probability of Default (PD)
tier_analysis['empirical_pd'] = tier_analysis['actual_defaults'] / tier_analysis['total_borrowers']


# --- IFRS-9 ACCOUNTING EXPECTED CREDIT LOSS (ECL) SIMULATION ---
# Operational assumptions: Flat $10k Exposure at Default (EAD) | 45% Loss Given Default (LGD) constant severity
EAD_per_borrower = 10000
LGD = 0.45

tier_analysis['EAD_portfolio'] = tier_analysis['total_borrowers'] * EAD_per_borrower
tier_analysis['Expected_Credit_Loss'] = tier_analysis['empirical_pd'] * LGD * tier_analysis['EAD_portfolio']

print("[DEBUG] Credit Score Stratification Performance Matrix:")
print(tier_analysis[['credit_tier', 'total_borrowers', 'empirical_pd', 'Expected_Credit_Loss']].to_string(index=False))
print("-" * 65)

# Aggregate forward-looking metrics across entire portfolio volume
total_exposure = tier_analysis['EAD_portfolio'].sum()
total_ecl = tier_analysis['Expected_Credit_Loss'].sum()
portfolio_loss_rate = (total_ecl / total_exposure) * 100

print(f"• Total Portfolio Asset Exposure (EAD) : ${total_exposure:,.2f}")
print(f"• Total Expected Credit Loss Asset (ECL): ${total_ecl:,.2f}")
print(f"• Combined Weighted Portfolio Loss Rate : {portfolio_loss_rate:.2f}%")

# Save structured analytics report to CSV
tier_analysis.to_csv('portfolio_metrics.csv', index=False)


# ==============================================================================
# AUTOMATED FINANCIAL VISUALIZATION FORECAST GENERATION
# ==============================================================================

# Chart A: Score Density Distribution Histogram
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='credit_score', bins=25, kde=True, color='royalblue', edgecolor='black')
plt.title('Credit Score Distribution across Portfolio', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Credit Score (Points)', fontsize=11, labelpad=10)
plt.ylabel('Borrower Count', fontsize=11, labelpad=10)
plt.xlim(450, 700)
plt.tight_layout()
plt.savefig('score_distribution.png', dpi=300)
plt.close()
print("\n[INFO] Distribution histogram exported: 'score_distribution.png'")

# Chart B: Monotonic Credit Score Validation
plt.figure(figsize=(10, 5))
tier_analysis['clean_tier'] = tier_analysis['credit_tier'].apply(lambda x: x.split('. ')[1])
ax = sns.barplot(
    data=tier_analysis, 
    x='clean_tier', 
    y='empirical_pd', 
    hue='clean_tier', 
    palette='coolwarm_r', 
    edgecolor='black', 
    legend=False
)
plt.title('Default Rate by Credit Score Tier (Monotonic Validation)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Credit Score Tier', fontsize=11, labelpad=10)
plt.ylabel('Observed Default Rate (PD)', fontsize=11, labelpad=10)
plt.ylim(0, 1.05)

# Insert calibrated data markers on top of bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height*100:.2f}%',
                    (p.get_x() + p.get_width() / 2., height + 0.02),
                    ha='center', va='center', fontsize=10, fontweight='bold', color='black',
                    xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('default_rate_by_tier.png', dpi=300)
plt.close()
print("[INFO] Monotonicity validation chart exported: 'default_rate_by_tier.png'")

print("-" * 65)
print("[SUCCESS] Financial dashboard generation and pipeline run execution complete\n")