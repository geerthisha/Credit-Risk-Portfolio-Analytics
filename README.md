# Credit Risk Portfolio Analytics Engine

An institutional-grade, forward-looking Expected Credit Loss (ECL) monitoring framework developed to simulate risk parameters across retail asset portfolios. The system interfaces an automated data ingestion pipeline with a native multi-page graphical user interface built via Tkinter, providing risk professionals with empirical insights regarding credit tier monotonicity, distribution profiling, and structural capital reserve adequacy.

---

## Commercial & Regulatory Context

Under post-crisis regulatory standards (IFRS 9 and CECL), financial institutions must transition away from legacy "incurred loss" accounting and instead compute forward-looking Expected Credit Losses (ECL) using statistical models. 

Managing an enterprise portfolio requires balancing two opposing pressures:
1. Under-reserving capital: Exposes the bank to structural insolvency, severe liquidity gaps, and regulatory breach if delinquency vectors spike.
2. Over-reserving capital: Traps tier-1 capital on the balance sheet unnecessarily, limiting the institution's capacity to deploy revenue-generating commercial loans.

This analytics engine resolves this operational friction. By translating raw financial tabular matrices into an interactive executive visualization plane, it allows risk managers to verify that internal credit ratings, score spreads, and impairment allocations accurately match empirical portfolio defaults.

---

## Core Portfolio Metrics

Automated extraction and dynamic aggregation from the underlying data matrices (portfolio_metrics.csv) establish the following balance sheet benchmarks:

* Total Assets Under Management (AUM / Exposure at Default): £167,130,000.00
* Expected Credit Loss (ECL) Impairment Allocation: £37,602,000.00
* Systemic Expected Portfolio Loss Rate: 22.50%

---

## Software Architecture & Design Features

* Asynchronous Data Ingestion Engine: Utilizes vectorized pandas routines to programmatically scan and clean production inputs, calculating rolling credit risk sums with zero execution bottlenecking.
* Synchronous Render Pipeline: Intentionally built around Tkinter's native single-threaded loop framework to guarantee layout stability on macOS, leveraging explicit layout parameters to frame interface objects.
* Proportional Geometry Management: Employs precise widget-packing padding controls to preserve a tight, intentional dark-mode layout container layout across view screens without risk of boundary distortion.
* Dynamic String Buffering: Leverages isolated directory structures (analyses) to handle contextual commentary on the fly, instantly mapping localized text outputs below high-resolution plots during screen navigation blocks.

---

## Quantitative Insights & Statistical Validation

### 1. Consumer Score Density Profile
The underlying asset distribution features a risk score spread peaking at a median of 606 points. The left-skewed shape validates that the credit scorecard logic successfully separates distinct consumer risk segments, concentrating high-risk or subprime borrowers into actionable pricing categories.

### 2. Empirical Monotonicity Verification
Quantitative risk models rely on the assumption of Monotonicity—as consumer credit grades decline across tiers, observed default probabilities must strictly increase. Empirical tracking inside this engine confirms an unbroken progression:
* Super Prime Tier Observed Default Risk: 6.74%
* Deep Subprime Tier Observed Default Risk: 94.38%

This sequential risk curve verifies the predictive power of the credit rating setup, mathematically validating the portfolio's £37,602,000.00 impairment reserve strategy.

---

## Installation & Production Deployment

### Prerequisites
* Python 3.10+ (Optimized for macOS execution architectures)
* Local Tkinter Tcl/Tk binaries
* Pip package manager

### 1. Environment Initialization
Clone the localized codebase and initialize an isolated workspace execution context:

```bash
git clone [https://github.com/your-username/credit-risk-engine.git](https://github.com/your-username/credit-risk-engine.git)
cd credit-risk-engine
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Dependency Resolution
Upgrade core packaging components and resolve mandatory analytical libraries:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install pandas pillow
```

### 3. Pipeline Execution
Ensure the source dataset (portfolio_metrics.csv) and corresponding static chart assets (score_distribution.png, default_rate_by_tier.png) are present in the project directory, then run:
```bash
python3 dashboard_popup.py
```

---

## Technology Stack

* **Programming Language:** Python 3.12+
* **Data Engineering Stack:** Pandas, NumPy
* **Imaging & Matrix Sampling Library:** Pillow (PIL)
* **Graphical Interface Infrastructure:** Tkinter Windowing Engine

---

## Project Repository Structure

```text
├── Credit Risk Benchmark Dataset.csv   # Raw baseline portfolio source input metrics
├── stage1_cleaning.py                  # Ingestion pipeline executing outlier clipping and imputation
├── stage2_features.py                  # Feature engineering module aggregating delinquency matrices
├── stage3_woe_iv.py                    # Weight of Evidence (WoE) transformation & Information Value ranking
├── stage4_pd_modeling.py               # Estimates Logistic Regression coefficients and AUC-ROC diagnostics
├── stage5_scorecard_scaling.py         # Calibrates regression logs into commercial points-based scorecards
├── stage6_ecl_dashboard.py             # Computes IFRS 9 Forward-Looking ECL capital simulations
├── score_distribution.png              # Output density report capturing consumer scorecard spreads
└── default_rate_by_tier.png            # Visual analytical validation confirming risk monotonicity
```
