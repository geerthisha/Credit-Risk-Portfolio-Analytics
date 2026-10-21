import numpy as np
import pandas as pd

from config import (
    CANDIDATE_MODEL_FEATURES,
    DELINQUENCY_COLUMNS,
    DELINQUENCY_FLAG,
    PROCESSED_DIR,
    PROJECT_DIR,
    TARGET,
)


OUTPUT_DIR = PROJECT_DIR / "outputs" / "stage3"
N_QUANTILE_BINS = 5
SMOOTHING = 0.5

DELINQUENCY_LABELS = ["0", "1", "2-3", "4-5", "6+"]


def validate_training_data(df):
    """Validate the training features and binary target."""
    required = CANDIDATE_MODEL_FEATURES + [TARGET]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("The training dataset is empty.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("The target contains invalid values.")

    if df[TARGET].nunique() != 2:
        raise ValueError("Both target classes are required.")

    for feature in CANDIDATE_MODEL_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise ValueError(f"{feature} must be numeric.")

        values = df[feature].dropna().to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"{feature} contains infinite values.")

    if not df[DELINQUENCY_FLAG].isin([0, 1]).all():
        raise ValueError("The delinquency flag must contain 0 or 1.")


def bin_feature(series, feature):
    """Create descriptive groups using training data only."""
    observed = series.dropna()

    if observed.empty:
        return (
            pd.Series("Missing", index=series.index),
            "all_missing",
        )

    if feature in DELINQUENCY_COLUMNS + ["total_delinquency"]:
        if (observed < 0).any():
            raise ValueError(f"{feature} contains negative counts.")

        if (observed % 1 != 0).any():
            raise ValueError(f"{feature} contains fractional counts.")

        grouped = pd.cut(
            series,
            bins=[-np.inf, 0, 1, 3, 5, np.inf],
            labels=DELINQUENCY_LABELS,
        )

        method = "fixed_delinquency_groups"

    elif observed.nunique() <= 10:
        grouped = series
        method = "individual_values"

    else:
        grouped = pd.qcut(
            series,
            q=N_QUANTILE_BINS,
            duplicates="drop",
        )

        method = "training_quantiles"

    labels = grouped.astype("string").fillna("Missing")

    return labels, method


def get_group_order(labels, method):
    """Return a readable ordering for the descriptive groups."""
    observed_labels = labels[labels != "Missing"].unique().tolist()

    if method == "fixed_delinquency_groups":
        order = [
            label
            for label in DELINQUENCY_LABELS
            if label in observed_labels
        ]

    elif method == "individual_values":
        order = sorted(observed_labels, key=float)

    elif method == "training_quantiles":
        def lower_boundary(label):
            return float(label[1:].split(",")[0])

        order = sorted(observed_labels, key=lower_boundary)

    else:
        order = observed_labels

    if labels.eq("Missing").any():
        order.append("Missing")

    return order


def calculate_woe_iv(labels, target, group_order):
    """Calculate smoothed WoE and IV for observed training groups."""
    working = pd.DataFrame({
        "group": labels,
        "target": target,
    })

    table = (
        working.groupby("group")["target"]
        .agg(records="size", events="sum")
        .reindex(group_order)
        .reset_index()
    )

    table["non_events"] = table["records"] - table["events"]
    table["event_rate"] = table["events"] / table["records"]
    table["sample_share"] = table["records"] / len(working)

    # Add the same pseudocount to each class in every observed group.
    number_of_groups = len(table)

    smoothed_event_total = (
        table["events"].sum() + SMOOTHING * number_of_groups
    )
    smoothed_non_event_total = (
        table["non_events"].sum() + SMOOTHING * number_of_groups
    )

    table["event_distribution"] = (
        table["events"] + SMOOTHING
    ) / smoothed_event_total

    table["non_event_distribution"] = (
        table["non_events"] + SMOOTHING
    ) / smoothed_non_event_total

    table["woe"] = np.log(
        table["non_event_distribution"]
        / table["event_distribution"]
    )

    table["iv_component"] = (
        table["non_event_distribution"]
        - table["event_distribution"]
    ) * table["woe"]

    if table["records"].sum() != len(working):
        raise RuntimeError("Group record counts do not reconcile.")

    if table["events"].sum() != target.sum():
        raise RuntimeError("Group event counts do not reconcile.")

    if not np.isclose(table["event_distribution"].sum(), 1.0):
        raise RuntimeError("Event distributions do not sum to one.")

    if not np.isclose(table["non_event_distribution"].sum(), 1.0):
        raise RuntimeError("Non-event distributions do not sum to one.")

    if not np.isfinite(
        table[["woe", "iv_component"]].to_numpy()
    ).all():
        raise RuntimeError("WoE or IV contains non-finite values.")

    return table, float(table["iv_component"].sum())


def main():
    print("[INFO] Stage 3: Training-only feature profiles and WoE/IV")

    train_df = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    validate_training_data(train_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ranking_rows = []
    combined_tables = []

    for feature in CANDIDATE_MODEL_FEATURES:
        labels, method = bin_feature(train_df[feature], feature)
        group_order = get_group_order(labels, method)

        table, iv = calculate_woe_iv(
            labels,
            train_df[TARGET],
            group_order,
        )

        table.insert(0, "feature", feature)
        table.insert(1, "binning_method", method)

        table.to_csv(
            OUTPUT_DIR / f"{feature}_woe_iv.csv",
            index=False,
        )

        combined_tables.append(table)

        ranking_rows.append({
            "feature": feature,
            "binning_method": method,
            "observed_groups": len(table),
            "missing_records": int(train_df[feature].isna().sum()),
            "minimum_group_records": int(table["records"].min()),
            "information_value": iv,
        })

    ranking = (
        pd.DataFrame(ranking_rows)
        .sort_values(
            ["information_value", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    ranking.to_csv(OUTPUT_DIR / "iv_ranking.csv", index=False)

    pd.concat(combined_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_feature_profiles.csv",
        index=False,
    )

    pd.DataFrame([{
        "training_records": len(train_df),
        "training_events": int(train_df[TARGET].sum()),
        "training_event_rate": float(train_df[TARGET].mean()),
        "features_profiled": len(ranking),
        "quantile_bins_requested": N_QUANTILE_BINS,
        "smoothing_per_class_per_group": SMOOTHING,
        "test_data_used": False,
    }]).to_csv(
        OUTPUT_DIR / "stage3_audit.csv",
        index=False,
    )

    print(f"[INFO] Training records: {len(train_df):,}")
    print(f"[INFO] Features profiled: {len(ranking)}")

    print("\n[INFO] Exploratory IV ranking:")
    print(
        ranking[
            ["rank", "feature", "information_value"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n[INFO] Positive WoE indicates lower relative event risk")
    print("[INFO] Missing values are reported as a separate group")
    print("[INFO] IV depends on the chosen grouping")
    print("[INFO] These tables do not select the final model features")
    print("[INFO] No test data was loaded")
    print(f"[SUCCESS] Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()