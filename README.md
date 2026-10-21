# Credit Risk Portfolio Analytics

**How effectively can logistic regression predict two-year serious delinquency from borrower characteristics in this benchmark dataset?**

A reproducible Python study of borrower risk, comparing logistic regression with a constant-probability baseline and gradient boosting. The project connects data validation, training-only exploration, cross-validated model selection, held-out evaluation, interpretable score scaling and a Tkinter dashboard.

## Results

The held-out test set contains **3,343 borrowers**, including **1,671 serious-delinquency events**.

| Model | Test ROC-AUC | 95% bootstrap interval | Average precision | Brier score ↓ | Log loss ↓ |
| --- | ---: | --- | ---: | ---: | ---: |
| Constant-probability baseline | 0.5000 | — | 0.4999 | 0.2500 | 0.6931 |
| Logistic regression | **0.8534** | **0.8402–0.8660** | **0.8544** | **0.1572** | **0.4841** |
| Gradient boosting | 0.8625 | 0.8496–0.8747 | 0.8596 | 0.1505 | 0.4643 |

Logistic regression provides useful discrimination within this benchmark and improves substantially on the baseline. Its training cross-validation ROC-AUC was **0.8478**, compared with **0.8534** on the test set.

Gradient boosting improves test ROC-AUC by **0.00912**, with a paired 95% bootstrap interval of **0.00469–0.01333**. It also achieves lower Brier score and log loss. This supports a small performance advantage, while logistic regression offers a directly additive score decomposition.

Intervals use 1,000 paired, class-stratified percentile bootstrap resamples. They describe test-sample uncertainty conditional on the fitted models, excluding training and model-selection uncertainty.

![Held-out ROC curves](outputs/stage5/test_roc.png)

## Risk concentration

Fixed probability bands summarise the frozen logistic model's predictions. Boundaries were chosen independently of test outcomes.

| Predicted-risk band | Borrowers | Mean predicted probability | Observed event rate | Share of illustrative loss |
| --- | ---: | ---: | ---: | ---: |
| 0–10% | 79 | 8.60% | 3.80% | 0.40% |
| 10–20% | 601 | 15.39% | 10.15% | 5.49% |
| 20–40% | 808 | 28.54% | 28.22% | 13.70% |
| 40–60% | 617 | 50.27% | 54.78% | 18.42% |
| 60–80% | 480 | 69.55% | 73.54% | 19.83% |
| 80–100% | 758 | 93.61% | 90.77% | 42.15% |

The highest-risk band contains **22.67% of borrowers and 42.15% of illustrative loss**. The highest two bands contain **37.03% of borrowers and 61.98% of illustrative loss**. Observed event rates increase across all six bands in this test sample; this is an empirical result, not an enforced model property.

![Borrower share and illustrative loss share](outputs/stage7/loss_concentration_by_band.png)

Illustrative loss is calculated as:

```text
sum(predicted two-year serious-delinquency probability × assumed EAD × assumed LGD)
```

Assumptions are **GBP 10,000 EAD per borrower** and **45% LGD**, with no discounting. Serious delinquency is used as a proxy for default. These assumptions produce GBP 7,575,638.75 of illustrative loss on GBP 33,430,000 of assumed exposure.

These amounts are not observed losses, representative lending-portfolio forecasts or IFRS 9 provisions. With constant EAD and LGD, loss shares are proportional to each band's sum of predicted probabilities. The currency is an illustrative choice; the source income currency is unspecified.

## Dataset and preparation

Source: [Credit Risk Benchmark Dataset](https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset), published on Kaggle by **adilshamim8**. The supplied dataset page lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution applies to the dataset; it does not establish a licence for this project's code.

The downloaded file contains 16,714 records, ten numerical predictors and a binary target. Exactly 8,357 records belong to each outcome class. The publisher does not explain the balancing procedure or earlier preprocessing. No missing values were detected in the supplied CSV.

Project modifications:

- Preserve the raw file unchanged; validate schema, values and the binary target.
- Require observed age of at least 18. No records were excluded by this rule. Retain the observed age of 101 because no documented upper eligibility limit is available.
- Remove two exact duplicate copies before splitting. Identical values could represent different borrowers; removal conservatively prevents copies crossing the split.
- Treat undocumented delinquency values 96 and 98 as unknown under a modelling assumption. After deduplication, 88 records contain these values; their indicators match across all three delinquency columns. Preserve one `delinquency_code_flag`.
- Create a stratified 80/20 split with seed 42: **13,369 training records and 3,343 test records**. Record IDs retain original CSV data-row positions and are excluded from modelling.

In training data, 69 of 73 flagged records experienced serious delinquency (94.52%), versus 49.75% among unflagged records. This association does not establish the meaning of the codes. A connection to other credit datasets has not been verified.

### Data dictionary

| Column | Publisher description / interpretation |
| --- | --- |
| `rev_util` | Revolving credit balance divided by credit limit |
| `age` | Borrower age in years |
| `late_30_59` | Count of occasions 30–59 days past due; historical window unspecified |
| `debt_ratio` | Debt relative to income or assets; denominator ambiguous |
| `monthly_inc` | Monthly income; currency and gross/net basis unspecified |
| `open_credit` | Number of open credit lines and loans |
| `late_90` | Count of occasions at least 90 days late; historical window unspecified |
| `real_estate` | Number of real-estate loans or credit lines |
| `late_60_89` | Count of occasions 60–89 days past due; historical window unspecified |
| `dependents` | Number of dependents |
| `dlq_2yrs` | Serious delinquency in the next two years: 1 = yes, 0 = no; precise severity threshold unspecified |

## Modelling approach

Exploratory WoE/IV tables use training data only, with a 0.5 pseudocount added to both classes in every observed group. These tables are descriptive and do not supply learned bins to cross-validation.

Model selection uses five identical stratified training folds. Median imputation, 99th-percentile upper caps for utilisation and debt ratio, and logistic standardisation are fitted within each fold. Income is compared in raw form and as `log1p(income)`. Structural unusual-code handling and deterministic feature engineering precede these learned transformations.

Two delinquency representations are compared: the separate counts, and a replacement composite `late_30_59 + 2×late_60_89 + 3×late_90`. The weights are a severity convention. The composite and all its components are not fitted together. Their best logistic CV AUCs are 0.8472 and 0.8478 respectively: little difference, with separate counts selected by mean CV AUC.

The selected logistic model uses separate counts, logged income and `C=10`, the best of 0.1, 1 and 10. Regularisation and income transformation had negligible effects on AUC across these candidates. The gradient-boosting comparison selects seven leaf nodes from seven and fifteen, with 150 boosting iterations and learning rate 0.1.

A threshold of **0.50** was fixed before initial test evaluation. Logistic precision is **78.91%**, recall **73.67%** and specificity **80.32%**. A lending cutoff would require representative data and explicit decision costs.

Models were frozen after initial held-out evaluation. Reporting stages load them without fitting or recalibration. Further modelling informed by these results would require fresh independent evaluation. A Git freeze tag should only be described as created once it has actually been committed.

## Score scaling and checks

Scores are derived directly from the frozen logistic pipeline:

```text
score = 600 − (20 / ln(2)) × model log-odds
```

600 points represents 1:1 non-event/event odds; 20 additional points doubles those odds. Higher scores mean lower predicted serious-delinquency risk. Unrounded scores preserve ROC-AUC; rounded scores are for display only.

Exports include feature contributions, imputation values, caps, processed training means/scales and a manual reconstruction formula. Logged income's multiplier applies to `ln(1 + income)`. Base points refer to the mean of processed predictors, which need not represent an actual borrower. Contributions are not causal effects or validated lending reason codes.

Checks cover probability round trips, score contributions, manual reconstruction, reference odds, unchanged AUC and agreement with evaluated predictions. The model's SHA-256 fingerprint is recorded. Manual reconstruction differed by at most approximately 1.7×10⁻¹³ points in the verified run.

## Limitations

- The approximately 50/50 sample is not established as representative of lending prevalence. Calibration, average precision, threshold metrics and absolute losses are sample-dependent.
- Predictor measurement dates, historical windows, sampling and earlier preprocessing are incompletely documented. The target's exact severity threshold is unspecified.
- Calibration differs by band. Logistic predictions overestimate observed risk in the 10–20% and 80–100% bands and underestimate it in several middle bands. The lowest band has 79 borrowers and only three events.
- Debt ratio has an ambiguous definition and extreme values. Its earlier negative adjusted linear coefficient contrasts with generally higher unadjusted event rates in upper ratio groups; no causal interpretation is claimed.
- Uncapped delinquency terms can extrapolate. The minimum score, 89.68, is driven primarily by 17 recorded 90+ day late-payment episodes against a training maximum of 13. There are 64 test scores below 400 (1.91%). Near-certain predicted risk is not independently validated.
- Random holdout evaluation does not demonstrate temporal, external-population or production performance. No regulatory compliance or deployment readiness is claimed.

![Held-out calibration](outputs/stage5/test_calibration.png)

## Setup and execution

Tested locally with **Python 3.14.2 on macOS**. Dependency versions are pinned in `requirements.txt`. Independent clean-environment reproduction remains to be checked.

```bash
git clone https://github.com/geerthisha/Credit-Risk-Portfolio-Analytics.git
cd Credit-Risk-Portfolio-Analytics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. Tkinter/Tcl-Tk must be available separately from pip. Check it with `python -m tkinter`; this should open a small test window. If unavailable, install Python with Tk support or the appropriate OS package. The analytical stages do not require a graphical display.

### First run from a fresh clone

Download the CSV from the linked Kaggle dataset page. Create `data/raw/` and place the file at exactly:

```text
data/raw/Credit Risk Benchmark Dataset.csv
```

Raw data, borrower-level exports and fitted models may be excluded from Git. Generate them before launching the dashboard:

```bash
python run_pipeline.py --rebuild
python -m unittest test_project -v
python dashboard_popup.py
```

`--rebuild` runs Stages 1–7 and overwrites generated models and outputs. Reproduce in a separate checkout or copy when preserving an existing frozen checkpoint. Rebuilt model bytes and results can vary across software or platform versions.

### Reporting with existing models

```bash
python run_pipeline.py
python dashboard_popup.py
```

The default runner executes Stages 5–7 using existing models and prepared test data. It stops on a failed stage and returns an error code. The dashboard is launched separately and reports missing files rather than substituting fixed figures.

## Project layout

```text
├── config.py
├── features.py
├── model_components.py
├── run_pipeline.py
├── stage1_cleaning.py
├── stage2_features.py
├── stage3_woe_iv.py
├── stage4_pd_modeling.py
├── stage5_evaluation.py
├── stage6_scorecard_scaling.py
├── stage7_portfolio_analysis.py
├── dashboard_popup.py
├── test_project.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── outputs/
    ├── stage3/
    ├── stage4/
    ├── stage5/
    ├── stage6/
    └── stage7/
```

The dashboard presents held-out performance, risk concentration, charts and limitations. Referenced chart PNGs and summary CSV/JSON outputs should be committed so results can be inspected without running the project.

## Validation status

Six automated tests passed in the author's working environment: split separation, valid probabilities and record coverage, score/contribution reconciliation, fresh-process model loading, training-learned caps and missing-preserving feature engineering. The reporting runner completed Stages 5–7 with unchanged evaluated metrics.

Clean-environment reproduction, final repository cleanup and a Git checkpoint remain release tasks. No checkpoint tag or independent reproduction is claimed here.

