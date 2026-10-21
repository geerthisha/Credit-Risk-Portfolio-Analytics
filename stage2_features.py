import numpy as np
import pandas as pd

from config import (
    CANDIDATE_MODEL_FEATURES,
    DELINQUENCY_COLUMNS,
    DELINQUENCY_FLAG,
    PROCESSED_DIR,
    RECORD_ID,
    SEPARATE_COUNT_FEATURES,
    TARGET,
)
from features import engineer_features


def validate_input(df, name):
    """Validate the Stage 1 export before feature engineering."""
    required = [RECORD_ID, TARGET] + SEPARATE_COUNT_FEATURES
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"{name} contains no records.")

    if df[RECORD_ID].isna().any():
        raise ValueError(f"{name} contains missing record IDs.")

    if df[RECORD_ID].duplicated().any():
        raise ValueError(f"{name} contains duplicate record IDs.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(f"{name} contains invalid target values.")

    if not df[DELINQUENCY_FLAG].isin([0, 1]).all():
        raise ValueError(f"{name} contains invalid flag values.")

    predictors = df[SEPARATE_COUNT_FEATURES]

    if not all(
        pd.api.types.is_numeric_dtype(predictors[column])
        for column in predictors.columns
    ):
        raise ValueError(f"{name} contains non-numeric predictors.")

    if np.isinf(predictors.to_numpy(dtype=float)).any():
        raise ValueError(f"{name} contains infinite predictor values.")


def validate_output(original, featured, name):
    """Check record preservation and weighted-total consistency."""
    if len(featured) != len(original):
        raise RuntimeError(f"{name}: row count changed.")

    if not featured[original.columns].equals(original):
        raise RuntimeError(f"{name}: original values changed.")

    component_missing = original[DELINQUENCY_COLUMNS].isna().any(
        axis=1
    )
    total_missing = featured["total_delinquency"].isna()

    if not component_missing.equals(total_missing):
        raise RuntimeError(
            f"{name}: missing delinquency components "
            "were not preserved in the total."
        )

    # This relationship is specific to the current benchmark exports.
    flagged = original[DELINQUENCY_FLAG].eq(1)

    if not total_missing.equals(flagged):
        raise RuntimeError(
            f"{name}: missing totals do not match the benchmark's "
            "unusual-code flag. Review the Stage 1 outputs."
        )

    values = featured["total_delinquency"].dropna()

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{name}: non-finite weighted totals.")

    if (values < 0).any():
        raise ValueError(f"{name}: negative weighted totals.")

    missing_features = (
        set(CANDIDATE_MODEL_FEATURES) - set(featured.columns)
    )

    if missing_features:
        raise RuntimeError(
            f"{name}: missing candidate features: "
            f"{sorted(missing_features)}"
        )


def build_training_profile(train_df):
    """Summarise training outcomes using fixed descriptive groups."""
    group_order = ["0", "1", "2-3", "4-5", "6+", "Unknown"]

    groups = pd.cut(
        train_df["total_delinquency"],
        bins=[-np.inf, 0, 1, 3, 5, np.inf],
        labels=group_order[:-1],
        right=True,
    )

    profile_df = pd.DataFrame({
        "delinquency_group": (
            groups.astype("string").fillna("Unknown")
        ),
        TARGET: train_df[TARGET],
    })

    profile = (
        profile_df.groupby("delinquency_group")[TARGET]
        .agg(
            records="size",
            events="sum",
            event_rate="mean",
        )
        .reindex(group_order)
        .dropna(subset=["records"])
        .reset_index()
    )

    profile[["records", "events"]] = profile[
        ["records", "events"]
    ].astype(int)

    if profile["records"].sum() != len(train_df):
        raise RuntimeError("Training profile row counts do not reconcile.")

    if profile["events"].sum() != train_df[TARGET].sum():
        raise RuntimeError("Training profile event counts do not reconcile.")

    return profile


def main():
    print("[INFO] Stage 2: Feature engineering")

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    validate_input(train_df, "Training data")
    validate_input(test_df, "Test data")

    if set(train_df[RECORD_ID]) & set(test_df[RECORD_ID]):
        raise ValueError("Training and test record IDs overlap.")

    train_features = engineer_features(train_df)
    test_features = engineer_features(test_df)

    validate_output(train_df, train_features, "Training data")
    validate_output(test_df, test_features, "Test data")

    training_profile = build_training_profile(train_features)

    training_summary = (
        train_features[CANDIDATE_MODEL_FEATURES]
        .describe(percentiles=[0.01, 0.50, 0.99])
        .T
    )
    training_summary["missing_count"] = train_features[
        CANDIDATE_MODEL_FEATURES
    ].isna().sum()
    training_summary.index.name = "feature"

    audit_rows = []

    for name, original, featured in [
        ("training", train_df, train_features),
        ("test", test_df, test_features),
    ]:
        audit_rows.append({
            "split": name,
            "input_records": len(original),
            "output_records": len(featured),
            "missing_weighted_totals": int(
                featured["total_delinquency"].isna().sum()
            ),
            "flagged_records": int(
                featured[DELINQUENCY_FLAG].sum()
            ),
            "original_values_preserved": (
                featured[original.columns].equals(original)
            ),
            "missing_totals_match_flag": (
                featured["total_delinquency"]
                .isna()
                .equals(featured[DELINQUENCY_FLAG].eq(1))
            ),
        })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_features.to_csv(
        PROCESSED_DIR / "train_features.csv",
        index=False,
    )
    test_features.to_csv(
        PROCESSED_DIR / "test_features.csv",
        index=False,
    )
    training_profile.to_csv(
        PROCESSED_DIR / "training_delinquency_profile.csv",
        index=False,
    )
    training_summary.to_csv(
        PROCESSED_DIR / "training_feature_summary.csv"
    )
    pd.DataFrame(audit_rows).to_csv(
        PROCESSED_DIR / "stage2_audit.csv",
        index=False,
    )

    print(f"[INFO] Training records: {len(train_features):,}")
    print(f"[INFO] Test records: {len(test_features):,}")

    for name, featured in [
        ("training", train_features),
        ("test", test_features),
    ]:
        missing_count = int(
            featured["total_delinquency"].isna().sum()
        )
        print(
            f"[INFO] Missing weighted totals in {name}: "
            f"{missing_count:,}"
        )

    print("[INFO] Missing totals match the benchmark code flag")

    print("\n[INFO] Training serious-delinquency rates by group:")
    print(
        training_profile.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n[INFO] Original predictors and code flag preserved")
    print("[INFO] No imputation, scaling or learned binning applied")
    print(f"[SUCCESS] Files saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()