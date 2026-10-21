import hashlib
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.special import expit
from sklearn.metrics import roc_auc_score

from config import (
    MODEL_DIR,
    PROCESSED_DIR,
    RECORD_ID,
    SCORE_POINTS_TO_DOUBLE_ODDS,
    SCORE_REFERENCE_NON_EVENT_ODDS,
    SCORE_REFERENCE_POINTS,
    SEPARATE_COUNT_FEATURES,
    STAGE4_OUTPUT_DIR,
    STAGE5_OUTPUT_DIR,
    STAGE6_OUTPUT_DIR,
    TARGET,
)


def file_sha256(path):
    """Calculate a fingerprint of a model file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_scaling_parameters():
    """Calculate the factor and offset for non-event/event odds."""
    if SCORE_POINTS_TO_DOUBLE_ODDS <= 0:
        raise ValueError("Points to double odds must be positive.")

    if SCORE_REFERENCE_NON_EVENT_ODDS <= 0:
        raise ValueError("Reference odds must be positive.")

    factor = SCORE_POINTS_TO_DOUBLE_ODDS / np.log(2)

    offset = (
        SCORE_REFERENCE_POINTS
        - factor * np.log(SCORE_REFERENCE_NON_EVENT_ODDS)
    )

    return factor, offset


def load_test_data():
    """Load and validate the held-out inputs."""
    df = pd.read_csv(PROCESSED_DIR / "test.csv")

    required = [RECORD_ID, TARGET] + SEPARATE_COUNT_FEATURES
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing test columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("The test dataset is empty.")

    if df[RECORD_ID].isna().any() or df[RECORD_ID].duplicated().any():
        raise ValueError("Test record IDs must be present and unique.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Invalid test target values.")

    if df[TARGET].nunique() != 2:
        raise ValueError("Both target classes are required.")

    return df


def build_reference_table(factor, offset):
    """Show the probability interpretation of selected score values."""
    probabilities = np.array([
        0.01,
        0.05,
        0.10,
        0.20,
        0.50,
        0.80,
        0.90,
        0.95,
        0.99,
    ])

    log_odds = np.log(probabilities / (1 - probabilities))
    scores = offset - factor * log_odds

    return pd.DataFrame({
        "event_probability": probabilities,
        "non_event_to_event_odds": (
            (1 - probabilities) / probabilities
        ),
        "score": scores,
    })


def check_evaluated_predictions(test_df, probabilities):
    """Confirm that predictions match the Stage 5 evaluation."""
    evaluated = pd.read_csv(
        STAGE5_OUTPUT_DIR / "test_predictions.csv"
    )

    comparison = test_df[[RECORD_ID, TARGET]].copy()
    comparison["current_probability"] = probabilities

    comparison = comparison.merge(
        evaluated[
            [RECORD_ID, TARGET, "logistic_regression_probability"]
        ],
        on=RECORD_ID,
        how="outer",
        suffixes=("_current", "_evaluated"),
        indicator=True,
        validate="one_to_one",
    )

    if not comparison["_merge"].eq("both").all():
        raise RuntimeError("Stage 5 and Stage 6 record IDs differ.")

    if not comparison[f"{TARGET}_current"].equals(
        comparison[f"{TARGET}_evaluated"]
    ):
        raise RuntimeError("Stage 5 and Stage 6 outcomes differ.")

    if not np.allclose(
        comparison["current_probability"],
        comparison["logistic_regression_probability"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("Predictions differ from held-out evaluation.")


def export_manual_scorecard(
    pipeline,
    feature_names,
    factor,
    offset,
    X,
):
    """Export fitted transformations and verify manual reconstruction."""
    preprocessing = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]

    imputer = preprocessing.named_steps["imputer"]
    capper = preprocessing.named_steps["capper"]
    income_transform = preprocessing.named_steps["income_transform"]
    scaler = preprocessing.named_steps["scaler"]

    coefficients = model.coef_[0]
    multipliers = -factor * coefficients / scaler.scale_

    manual_intercept = float(
        offset
        - factor * model.intercept_[0]
        - np.dot(multipliers, scaler.mean_)
    )

    parameter_rows = []

    for index, feature in enumerate(feature_names):
        log_applied = bool(
            income_transform.enabled
            and index == income_transform.column_index
        )
        cap = capper.caps_[index]

        parameter_rows.append({
            "feature": feature,
            "imputation_value": float(imputer.statistics_[index]),
            "upper_cap_applied": bool(np.isfinite(cap)),
            "upper_cap": (
                float(cap) if np.isfinite(cap) else np.nan
            ),
            "transformation": (
                "log1p" if log_applied else "identity"
            ),
            "processed_training_mean": float(scaler.mean_[index]),
            "processed_training_scale": float(scaler.scale_[index]),
            "logistic_coefficient": float(coefficients[index]),
            "points_per_processed_unit": float(multipliers[index]),
        })

    engineered = pipeline.named_steps["engineering"].transform(X)
    manual_values = []

    for row in parameter_rows:
        values = engineered[row["feature"]].to_numpy(
            dtype=float,
            copy=True,
        )

        values = np.where(
            np.isnan(values),
            row["imputation_value"],
            values,
        )

        if row["upper_cap_applied"]:
            values = np.minimum(values, row["upper_cap"])

        if row["transformation"] == "log1p":
            values = np.log1p(values)

        manual_values.append(values)

    manual_matrix = np.column_stack(manual_values)
    manual_scores = manual_intercept + manual_matrix @ multipliers

    pipeline_scores = (
        offset - factor * pipeline.decision_function(X)
    )

    if not np.allclose(
        manual_scores,
        pipeline_scores,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError("Manual score reconstruction failed.")

    maximum_error = float(
        np.max(np.abs(manual_scores - pipeline_scores))
    )

    pd.DataFrame(parameter_rows).to_csv(
        STAGE6_OUTPUT_DIR / "manual_scorecard_parameters.csv",
        index=False,
    )

    instructions = {
        "manual_intercept": manual_intercept,
        "formula": (
            "score = manual_intercept + sum("
            "points_per_processed_unit * processed_feature_value)"
        ),
        "input_requirement": (
            "Inputs must first receive the same Stage 1 unusual-code "
            "handling and flag construction."
        ),
        "feature_engineering": (
            "For the composite representation only: total_delinquency "
            "= late_30_59 + 2*late_60_89 + 3*late_90. "
            "Missing components produce a missing total."
        ),
        "processing_order": [
            "Replace missing feature values with exported imputation values.",
            "Apply each exported upper cap where indicated.",
            "Apply log1p where indicated, using the natural logarithm.",
            "Multiply processed values by their exported points multipliers.",
            "Sum contributions and add the manual intercept.",
        ],
        "income_interpretation": (
            "When income uses log1p, its multiplier applies to "
            "ln(1 + monthly_inc), not directly to raw income."
        ),
        "reference_interpretation": (
            "The original base_points is the score at the mean of the "
            "processed training predictors. This combination need not "
            "represent an actual borrower."
        ),
        "contribution_interpretation": (
            "Centred feature contributions describe score changes "
            "relative to processed training means. They are model "
            "contributions, not validated lending reason codes."
        ),
        "probability_formula": (
            "event_probability = expit((offset - score) / factor)"
        ),
        "factor": float(factor),
        "offset": float(offset),
        "manual_reconstruction_passed": True,
        "maximum_absolute_reconstruction_error": maximum_error,
    }

    with (
        STAGE6_OUTPUT_DIR / "manual_scorecard_instructions.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(instructions, file, indent=2)

    return manual_intercept, maximum_error


def main():
    print("[INFO] Stage 6: Frozen logistic score scaling")

    model_path = MODEL_DIR / "selected_logistic_pipeline.joblib"
    fingerprint_before = file_sha256(model_path)
    pipeline = joblib.load(model_path)

    with (
        STAGE4_OUTPUT_DIR / "model_selection.json"
    ).open(encoding="utf-8") as file:
        selection = json.load(file)

    feature_names = selection["selected_logistic_features"]

    test_df = load_test_data()
    X = test_df[SEPARATE_COUNT_FEATURES].copy()

    factor, offset = get_scaling_parameters()

    if not np.array_equal(pipeline.classes_, [0, 1]):
        raise ValueError("Expected binary classes ordered as [0, 1].")

    log_odds = np.asarray(pipeline.decision_function(X))
    probabilities = pipeline.predict_proba(X)[:, 1]

    if not np.isfinite(log_odds).all():
        raise ValueError("Model log-odds contain non-finite values.")

    if not np.allclose(
        expit(log_odds),
        probabilities,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("Log-odds and model probabilities disagree.")

    scores = offset - factor * log_odds

    reconstructed_probabilities = expit(
        (offset - scores) / factor
    )

    if not np.allclose(
        reconstructed_probabilities,
        probabilities,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("Score-to-probability conversion failed.")

    check_evaluated_predictions(test_df, probabilities)

    processed = np.asarray(pipeline[:-1].transform(X))
    model = pipeline.named_steps["model"]
    coefficients = model.coef_[0]

    if processed.shape[1] != len(feature_names):
        raise RuntimeError("Processed feature names do not align.")

    selected_columns = list(
        pipeline.named_steps["selection"].transformers_[0][2]
    )

    if selected_columns != feature_names:
        raise RuntimeError("Saved feature order differs from metadata.")

    base_points = float(
        offset - factor * model.intercept_[0]
    )
    contributions = -factor * processed * coefficients

    reconstructed_scores = (
        base_points + contributions.sum(axis=1)
    )

    if not np.allclose(
        reconstructed_scores,
        scores,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError("Feature contributions do not reconcile.")

    reference_log_odds = -np.log(
        SCORE_REFERENCE_NON_EVENT_ODDS
    )

    if not np.isclose(
        offset - factor * reference_log_odds,
        SCORE_REFERENCE_POINTS,
    ):
        raise RuntimeError("Reference score check failed.")

    probe_log_odds = (
        offset - SCORE_REFERENCE_POINTS
    ) / factor

    shifted_log_odds = (
        offset
        - SCORE_REFERENCE_POINTS
        - SCORE_POINTS_TO_DOUBLE_ODDS
    ) / factor

    odds_multiplier = np.exp(
        probe_log_odds - shifted_log_odds
    )

    if not np.isclose(odds_multiplier, 2.0):
        raise RuntimeError("Points-to-double-odds check failed.")

    y = test_df[TARGET].to_numpy()
    probability_auc = roc_auc_score(y, probabilities)
    score_auc = roc_auc_score(y, -scores)

    if not np.isclose(probability_auc, score_auc):
        raise RuntimeError("Score scaling changed ROC-AUC.")

    STAGE6_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manual_intercept, manual_error = export_manual_scorecard(
        pipeline=pipeline,
        feature_names=feature_names,
        factor=factor,
        offset=offset,
        X=X,
    )

    scored_records = test_df[[RECORD_ID, TARGET]].copy()
    scored_records["predicted_serious_delinquency"] = probabilities
    scored_records["credit_score"] = scores
    scored_records["credit_score_rounded"] = np.rint(
        scores
    ).astype(int)

    scored_records.to_csv(
        STAGE6_OUTPUT_DIR / "test_scores.csv",
        index=False,
    )

    contribution_table = pd.DataFrame(
        contributions,
        columns=[
            f"{feature}_points"
            for feature in feature_names
        ],
    )
    contribution_table.insert(
        0,
        RECORD_ID,
        test_df[RECORD_ID].to_numpy(),
    )
    contribution_table["base_points"] = base_points
    contribution_table["total_score"] = scores

    contribution_table.to_csv(
        STAGE6_OUTPUT_DIR / "test_score_contributions.csv",
        index=False,
    )

    income_logged = selection["income_log1p_selected"]

    terms = [
        "log1p(monthly_inc)"
        if feature == "monthly_inc" and income_logged
        else feature
        for feature in feature_names
    ]

    pd.DataFrame({
        "feature": feature_names,
        "model_term": terms,
        "logistic_coefficient": coefficients,
        "points_per_standardised_unit": -factor * coefficients,
    }).to_csv(
        STAGE6_OUTPUT_DIR / "scorecard_parameters.csv",
        index=False,
    )

    build_reference_table(factor, offset).to_csv(
        STAGE6_OUTPUT_DIR / "score_probability_reference.csv",
        index=False,
    )

    below_400 = int((scores < 400).sum())
    share_below_400 = below_400 / len(scores)

    # Save low-score diagnostics without changing the fitted model.
    diagnostics = scored_records.merge(
        test_df[[RECORD_ID] + SEPARATE_COUNT_FEATURES],
        on=RECORD_ID,
        validate="one_to_one",
    ).merge(
        contribution_table.drop(columns="total_score"),
        on=RECORD_ID,
        validate="one_to_one",
    )

    diagnostics.nsmallest(10, "credit_score").to_csv(
        STAGE6_OUTPUT_DIR / "lowest_score_diagnostics.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        scores,
        bins=30,
        color="royalblue",
        edgecolor="white",
    )
    ax.set(
        xlabel="Project-defined credit score",
        ylabel="Borrower count",
        title="Held-out logistic score distribution",
    )

    fig.text(
        0.5,
        0.02,
        (
            f"Unclipped scores; {below_400:,} of {len(scores):,} "
            f"borrowers ({share_below_400:.2%}) score below 400."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(
        STAGE6_OUTPUT_DIR / "test_score_distribution.png",
        dpi=200,
    )
    plt.close(fig)

    fingerprint_after = file_sha256(model_path)

    if fingerprint_before != fingerprint_after:
        raise RuntimeError("The model file changed during execution.")

    audit = {
        "test_records": len(test_df),
        "model_file": model_path.name,
        "model_sha256": fingerprint_before,
        "model_file_unchanged_during_run": True,
        "model_refitted": False,
        "reference_score": SCORE_REFERENCE_POINTS,
        "reference_non_event_to_event_odds": (
            SCORE_REFERENCE_NON_EVENT_ODDS
        ),
        "points_to_double_non_event_odds": (
            SCORE_POINTS_TO_DOUBLE_ODDS
        ),
        "factor": float(factor),
        "offset": float(offset),
        "base_points": base_points,
        "manual_intercept": manual_intercept,
        "manual_score_reconstruction_passed": True,
        "maximum_manual_reconstruction_error": manual_error,
        "score_direction": (
            "Higher score means lower predicted event risk"
        ),
        "score_minimum": float(scores.min()),
        "score_median": float(np.median(scores)),
        "score_maximum": float(scores.max()),
        "scores_below_400": below_400,
        "share_below_400": share_below_400,
        "probabilities_match_stage5": True,
        "score_probability_conversion_passed": True,
        "contributions_reconcile": True,
        "reference_score_check_passed": True,
        "points_to_double_odds_check_passed": True,
        "test_probability_auc": float(probability_auc),
        "test_negative_score_auc": float(score_auc),
        "rounding_policy": (
            "Rounded scores are for display only. "
            "Analysis uses unrounded scores."
        ),
        "interpretation": (
            "Scores represent serious-delinquency risk in the "
            "benchmark sample; they are not commercial credit ratings."
        ),
    }

    with (
        STAGE6_OUTPUT_DIR / "score_scaling_audit.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)

    print(f"[INFO] Test records scored: {len(test_df):,}")
    print(f"[INFO] Scaling factor: {factor:.4f}")
    print(f"[INFO] Scaling offset: {offset:.4f}")
    print(f"[INFO] Median score: {np.median(scores):.2f}")
    print(
        f"[INFO] Score range: "
        f"{scores.min():.2f} to {scores.max():.2f}"
    )
    print(
        f"[INFO] Scores below 400: "
        f"{below_400:,} ({share_below_400:.2%})"
    )
    print("[INFO] Predictions match Stage 5 evaluation")
    print("[INFO] Score contributions reconcile")
    print("[INFO] Manual score reconstruction passed")
    print(
        f"[INFO] Maximum manual reconstruction error: "
        f"{manual_error:.3e}"
    )
    print("[INFO] Reference score and doubling-odds checks passed")
    print("[INFO] Unrounded score ROC-AUC is unchanged")
    print("[INFO] Frozen model file was not modified")
    print(f"[SUCCESS] Files saved to: {STAGE6_OUTPUT_DIR}")


if __name__ == "__main__":
    main()