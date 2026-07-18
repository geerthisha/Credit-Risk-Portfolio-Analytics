import subprocess
import os

print("🚀 STARTING CREDIT RISK SCORECARD PIPELINE 🚀\n")

stages = [
    "stage1_cleaning.py",
    "stage2_features.py",
    "stage3_woe_iv.py",
    "stage4_pd_modeling.py",
    "stage5_scorecard_scaling.py",
    "stage6_ecl_dashboard.py",
    "dashboard_popup.py"
]

for stage in stages:
    print(f"⏳ Running: {stage}...")
    # This runs the terminal command for each script automatically
    result = subprocess.run(["python3", stage], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Completed: {stage}\n")
    else:
        print(f"❌ Failed: {stage}")
        print(result.stderr)
        break

print("🎉 pipeline execution fully complete! All data and charts are generated.")