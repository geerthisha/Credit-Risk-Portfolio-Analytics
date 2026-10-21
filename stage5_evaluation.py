import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import (
    BOOTSTRAP_REPETITIONS,
    CALIBRATION_BINS,
    EVALUATION_THRESHOLD,
    MODEL_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RECORD_ID,
    SEPARATE_COUNT_FEATURES,
    STAGE4_OUTPUT_DIR,
    STAGE5_OUTPUT_DIR,
    TARGET,
)


def load_evaluation_inputs():
    """Load held-out records and check split membership."""
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    train_ids = pd.read_csv(
        PROCESSED_DIR / "train.csv",
        usecols=[RECORD_ID],
    )[RECORD_ID]

    required = [RECORD_ID, TARGET] + SEPARATE_COUNT_FEATURES
    missing = set(required) - set(test_df.columns)

    if missing:
        raise ValueError(f"Missing test columns: {sorted(missing)}")

    if test_df.empty:
        raise ValueError("The test dataset is empty.")

    if (
        test_df[RECORD_ID].isna().any()
        or test_df[RECORD_ID].duplicated().any()
    ):
        raise ValueError("Test record IDs must be present and unique.")

    if set(test_df[RECORD_ID]) & set(train_ids):
        raise ValueError("Training and test record IDs overlap.")

    if not test_df[TARGET].isin([0, 1]).all():
        raise ValueError("Invalid test target values.")

    if test_df[TARGET].nunique() != 2:
        raise ValueError("Both test outcome classes are required.")

    X = test_df[SEPARATE_COUNT_FEATURES].copy()

    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            raise ValueError(f"{column} must be numeric.")

    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Test predictors contain infinite values.")

    y = test_df[TARGET].to_numpy(dtype=int)

    return test_df, X, y


def predict_event_probability(model, X):
    """Extract and validate probabilities for outcome 1."""
    classes = np.asarray(model.classes_)
    event_positions = np.flatnonzero(classes == 1)

    if len(event_positions) != 1:
        raise ValueError("The model does not identify event class 1.")

    probabilities = model.predict_proba(X)[:, event_positions[0]]

    if len(probabilities) != len(X):
        raise ValueError("Prediction and record counts differ.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain non-finite values.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predictions must be between zero and one.")

    return probabilities


def bootstrap_auc(y, predictions):
    """Use shared class-stratified resamples for all frozen models."""
    rng = np.random.default_rng(RANDOM_STATE)

    non_events = np.flatnonzero(y == 0)
    events = np.flatnonzero(y == 1)

    samples = {name: [] for name in predictions}

    for _ in range(BOOTSTRAP_REPETITIONS):
        indices = np.concatenate([
            rng.choice(non_events, len(non_events), replace=True),
            rng.choice(events, len(events), replace=True),
        ])

        for name, probabilities in predictions.items():
            samples[name].append(
                roc_auc_score(y[indices], probabilities[indices])
            )

    return {
        name: np.asarray(values)
        for name, values in samples.items()
    }


def calculate_metrics(name, y, probabilities, auc_samples):
    """Calculate probability metrics and fixed-threshold results."""
    predicted = (probabilities >= EVALUATION_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predicted,
        labels=[0, 1],
    ).ravel()

    lower, upper = np.quantile(auc_samples, [0.025, 0.975])

    return {
        "model": name,
        "test_records": len(y),
        "event_rate": float(y.mean()),
        "roc_auc": roc_auc_score(y, probabilities),
        "roc_auc_ci_lower": float(lower),
        "roc_auc_ci_upper": float(upper),
        "average_precision": average_precision_score(y, probabilities),
        "brier_score": brier_score_loss(y, probabilities),
        "log_loss": log_loss(y, probabilities, labels=[0, 1]),
        "mean_predicted_probability": float(probabilities.mean()),
        "threshold": EVALUATION_THRESHOLD,
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "specificity": tn / (tn + fp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def build_calibration_table(y, probabilities):
    """Summarise outcomes within fixed probability intervals."""
    edges = np.linspace(0, 1, CALIBRATION_BINS + 1)

    data = pd.DataFrame({
        "probability": probabilities,
        "outcome": y,
        "probability_bin": pd.cut(
            probabilities,
            bins=edges,
            include_lowest=True,
        ),
    })

    table = (
        data.groupby("probability_bin", observed=True)
        .agg(
            records=("outcome", "size"),
            events=("outcome", "sum"),
            mean_predicted_probability=("probability", "mean"),
            observed_event_rate=("outcome", "mean"),
        )
        .reset_index()
    )

    if table["records"].sum() != len(y):
        raise RuntimeError("Calibration record counts do not reconcile.")

    if table["events"].sum() != y.sum():
        raise RuntimeError("Calibration event counts do not reconcile.")

    return table


def save_evaluation_plots(y, predictions, calibration_tables):
    """Save held-out ROC, precision-recall and calibration plots."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, probabilities in predictions.items():
        fpr, tpr, _ = roc_curve(y, probabilities)

        ax.plot(
            fpr,
            tpr,
            label=f"{name}: AUC={roc_auc_score(y, probabilities):.3f}",
        )

    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Held-out ROC curves",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(STAGE5_OUTPUT_DIR / "test_roc.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))

    for name, probabilities in predictions.items():
        precision, recall, _ = precision_recall_curve(y, probabilities)
        ax.plot(recall, precision, label=name)

    ax.axhline(
        y.mean(),
        linestyle="--",
        color="grey",
        label="Test event rate",
    )
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Held-out precision-recall curves",
        ylim=(0, 1.05),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        STAGE5_OUTPUT_DIR / "test_precision_recall.png",
        dpi=200,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))

    for name, table in calibration_tables.items():
        ax.plot(
            table["mean_predicted_probability"],
            table["observed_event_rate"],
            marker="o",
            label=name,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="grey",
        label="Perfect calibration",
    )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed event rate",
        title="Held-out calibration: balanced benchmark sample",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        STAGE5_OUTPUT_DIR / "test_calibration.png",
        dpi=200,
    )
    plt.close(fig)


def main():
    print("[INFO] Stage 5: Frozen-model held-out evaluation")

    if not 0 < EVALUATION_THRESHOLD < 1:
        raise ValueError("The evaluation threshold must be in (0, 1).")

    if BOOTSTRAP_REPETITIONS < 1:
        raise ValueError("At least one bootstrap repetition is required.")

    if CALIBRATION_BINS < 1:
        raise ValueError("At least one calibration bin is required.")

    test_df, X, y = load_evaluation_inputs()

    with (
        STAGE4_OUTPUT_DIR / "model_selection.json"
    ).open(encoding="utf-8") as file:
        selection = json.load(file)

    comparison_files = {}

    for name in ["naive_baseline", "gradient_boosting"]:
        matches = [
            experiment
            for experiment in selection["experiments"]
            if experiment["model"] == name
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected one saved comparison model for {name}."
            )

        comparison_files[name] = matches[0]["model_file"]

    model_files = {
        "naive_baseline": comparison_files["naive_baseline"],
        "logistic_regression": "selected_logistic_pipeline.joblib",
        "gradient_boosting": comparison_files["gradient_boosting"],
    }

    predictions = {}

    for name, filename in model_files.items():
        model = joblib.load(MODEL_DIR / filename)
        predictions[name] = predict_event_probability(model, X)

        print(f"[INFO] Generated frozen predictions: {name}")

    print("[INFO] Calculating bootstrap AUC intervals")
    auc_samples = bootstrap_auc(y, predictions)

    STAGE5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = []
    calibration_tables = {}
    prediction_table = test_df[[RECORD_ID, TARGET]].copy()

    for name, probabilities in predictions.items():
        metrics.append(
            calculate_metrics(
                name,
                y,
                probabilities,
                auc_samples[name],
            )
        )

        prediction_table[f"{name}_probability"] = probabilities

        calibration = build_calibration_table(y, probabilities)
        calibration_tables[name] = calibration

        calibration.to_csv(
            STAGE5_OUTPUT_DIR / f"{name}_calibration.csv",
            index=False,
        )

    metrics_table = pd.DataFrame(metrics)

    metrics_table.to_csv(
        STAGE5_OUTPUT_DIR / "test_metrics.csv",
        index=False,
    )
    prediction_table.to_csv(
        STAGE5_OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # Shared resamples preserve pairing between model predictions.
    bootstrap_differences = (
        auc_samples["gradient_boosting"]
        - auc_samples["logistic_regression"]
    )

    difference_lower, difference_upper = np.quantile(
        bootstrap_differences,
        [0.025, 0.975],
    )

    auc_difference = (
        roc_auc_score(y, predictions["gradient_boosting"])
        - roc_auc_score(y, predictions["logistic_regression"])
    )

    pd.DataFrame([{
        "comparison": "gradient_boosting_minus_logistic",
        "test_auc_difference": float(auc_difference),
        "ci_lower": float(difference_lower),
        "ci_upper": float(difference_upper),
    }]).to_csv(
        STAGE5_OUTPUT_DIR / "paired_auc_comparison.csv",
        index=False,
    )

    save_evaluation_plots(y, predictions, calibration_tables)

    audit = {
        "test_records": len(y),
        "test_events": int(y.sum()),
        "test_event_rate": float(y.mean()),
        "model_files": model_files,
        "selected_logistic_parameters": selection[
            "selected_logistic_parameters"
        ],
        "selected_logistic_feature_set": selection[
            "selected_logistic_feature_set"
        ],
        "threshold": EVALUATION_THRESHOLD,
        "threshold_policy": (
            "Fixed before initial test evaluation; not tuned on test data."
        ),
        "threshold_interpretation": (
            "Benchmark midpoint, not a validated lending decision policy."
        ),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_method": "paired class-stratified percentile",
        "confidence_level": 0.95,
        "calibration_bins": CALIBRATION_BINS,
        "random_state": RANDOM_STATE,
        "models_refitted": False,
        "model_freeze_policy": (
            "Retain the evaluated models. Further development informed "
            "by test results requires fresh independent evaluation."
        ),
        "interval_scope": (
            "Test-sample uncertainty conditional on fitted models; "
            "excludes training and model-selection uncertainty."
        ),
    }

    with (
        STAGE5_OUTPUT_DIR / "evaluation_audit.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)

    print("\n[INFO] Held-out results:")
    print(
        metrics_table[
            [
                "model",
                "roc_auc",
                "roc_auc_ci_lower",
                "roc_auc_ci_upper",
                "average_precision",
                "brier_score",
                "log_loss",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        "\n[INFO] Gradient boosting minus logistic test ROC-AUC: "
        f"{auc_difference:.4f}"
    )
    print(
        "[INFO] Paired 95% bootstrap interval: "
        f"[{difference_lower:.4f}, {difference_upper:.4f}]"
    )

    print(
        f"\n[INFO] Classification threshold: "
        f"{EVALUATION_THRESHOLD:.2f}"
    )
    print("[INFO] Threshold fixed before initial test evaluation")
    print("[INFO] Models were loaded without refitting")
    print("[INFO] Evaluated model settings remain frozen")
    print("[INFO] Calibration describes this benchmark sample")
    print(f"[SUCCESS] Results saved to: {STAGE5_OUTPUT_DIR}")


if __name__ == "__main__":
    main()