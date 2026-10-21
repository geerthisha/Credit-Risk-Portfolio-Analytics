import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    COUNT_COLUMNS,
    DELINQUENCY_COLUMNS,
    DELINQUENCY_FLAG,
    MINIMUM_AGE,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA_FILE,
    RAW_PREDICTORS,
    RECORD_ID,
    SUSPICIOUS_DELINQUENCY_VALUES,
    TARGET,
    TEST_SIZE,
)


def load_and_validate_data(path):
    """Load the data and validate its schema and values."""
    df = pd.read_csv(path)

    # Preserve a possible source identifier as metadata.
    source_id = None
    if "Unnamed: 0" in df.columns:
        source_id = df.pop("Unnamed: 0").copy()

    required = RAW_PREDICTORS + [TARGET]
    missing = set(required) - set(df.columns)
    unexpected = set(df.columns) - set(required)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if unexpected:
        raise ValueError(f"Unexpected columns: {sorted(unexpected)}")

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if np.isinf(df.to_numpy(dtype=float)).any():
        raise ValueError("The dataset contains infinite values.")

    if df[TARGET].isna().any():
        raise ValueError("The target contains missing values.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("The target must contain both 0 and 1 only.")

    non_age_predictors = [
        column for column in RAW_PREDICTORS if column != "age"
    ]

    if (df[non_age_predictors] < 0).any().any():
        raise ValueError("Non-age predictors contain negative values.")

    for column in COUNT_COLUMNS:
        values = df[column].dropna()

        if (values % 1 != 0).any():
            raise ValueError(f"{column} contains non-integer counts.")

    df[TARGET] = df[TARGET].astype(int)

    # Record identifiers refer to data-row positions in the original CSV.
    df.insert(0, RECORD_ID, df.index.to_numpy() + 1)

    if source_id is not None:
        df.insert(1, "source_id", source_id)

    return df


def prepare_data(df):
    """Apply eligibility rules, remove duplicates and flag unusual codes."""
    df = df.copy()
    input_records = len(df)

    missing_age = df["age"].isna()
    underage = df["age"].lt(MINIMUM_AGE)
    eligible = ~(missing_age | underage)

    missing_age_count = int(missing_age.sum())
    underage_count = int(underage.sum())

    print(f"[INFO] Excluded for missing age: {missing_age_count:,}")
    print(
        f"[INFO] Excluded for age below {MINIMUM_AGE}: "
        f"{underage_count:,}"
    )

    # No upper age cutoff is imposed; age 101 is not inherently invalid.
    df = df.loc[eligible].copy()

    # Identifiers are excluded from duplicate comparisons.
    duplicate_columns = RAW_PREDICTORS + [TARGET]
    duplicate_mask = df.duplicated(subset=duplicate_columns)
    duplicates_removed = int(duplicate_mask.sum())

    print(
        f"[INFO] Exact duplicate copies removed: "
        f"{duplicates_removed:,}"
    )

    df = df.loc[~duplicate_mask].copy()

    unusual = df[DELINQUENCY_COLUMNS].isin(
        SUSPICIOUS_DELINQUENCY_VALUES
    )

    flags_match = bool(
        unusual.eq(unusual.iloc[:, 0], axis=0).all().all()
    )

    print(f"[INFO] Delinquency-code flags identical: {flags_match}")

    # Values 96 and 98 are treated as unknown under a modelling assumption.
    df[DELINQUENCY_FLAG] = unusual.any(axis=1).astype(int)

    unusual_counts = {}

    for column in DELINQUENCY_COLUMNS:
        count = int(unusual[column].sum())
        unusual_counts[f"{column}_unusual_count"] = count

        print(f"[INFO] {column}: {count:,} unusual values")

        df.loc[unusual[column], column] = np.nan

    audit = {
        "input_records": input_records,
        "minimum_eligible_age": MINIMUM_AGE,
        "upper_age_limit_applied": False,
        "excluded_missing_age": missing_age_count,
        "excluded_under_minimum_age": underage_count,
        "duplicate_copies_removed": duplicates_removed,
        "retained_records": len(df),
        "delinquency_flags_identical": flags_match,
        "records_with_unusual_delinquency_codes": int(
            df[DELINQUENCY_FLAG].sum()
        ),
        **unusual_counts,
    }

    expected_records = (
        input_records
        - missing_age_count
        - underage_count
        - duplicates_removed
    )

    if len(df) != expected_records:
        raise RuntimeError("Preparation row counts do not reconcile.")

    return df, audit


def print_split_summary(name, df):
    """Print record counts and the serious-delinquency event rate."""
    events = int(df[TARGET].sum())
    event_rate = float(df[TARGET].mean())

    print(
        f"[INFO] {name}: {len(df):,} records | "
        f"{events:,} events | event rate {event_rate:.4%}"
    )


def main():
    print("[INFO] Stage 1: Data validation and splitting")

    raw_df = load_and_validate_data(RAW_DATA_FILE)
    print(f"[INFO] Input records: {len(raw_df):,}")

    prepared_df, audit = prepare_data(raw_df)

    if set(prepared_df[TARGET].unique()) != {0, 1}:
        raise ValueError(
            "Both target classes must remain after preparation."
        )

    train_df, test_df = train_test_split(
        prepared_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=prepared_df[TARGET],
    )

    train_ids = set(train_df[RECORD_ID])
    test_ids = set(test_df[RECORD_ID])

    if train_ids & test_ids:
        raise RuntimeError("Training and test record IDs overlap.")

    if len(train_df) + len(test_df) != len(prepared_df):
        raise RuntimeError("Split row counts do not reconcile.")

    print_split_summary("Training", train_df)
    print_split_summary("Test", test_df)

    # Only training outcomes are used for the unusual-code feature audit.
    code_summary = (
        train_df.groupby(DELINQUENCY_FLAG)[TARGET]
        .agg(
            records="size",
            events="sum",
            event_rate="mean",
        )
        .reset_index()
    )

    print("\n[INFO] Unusual-code groups in training data:")
    print(
        code_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    audit.update({
        "training_records": len(train_df),
        "training_events": int(train_df[TARGET].sum()),
        "training_event_rate": float(train_df[TARGET].mean()),
        "test_records": len(test_df),
        "test_events": int(test_df[TARGET].sum()),
        "test_event_rate": float(test_df[TARGET].mean()),
        "record_id_overlap": len(train_ids & test_ids),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    code_summary.to_csv(
        PROCESSED_DIR / "training_delinquency_code_summary.csv",
        index=False,
    )

    pd.DataFrame([audit]).to_csv(
        PROCESSED_DIR / "stage1_audit.csv",
        index=False,
    )

    print("\n[INFO] Imputation and capping deferred to modelling")
    print(f"[SUCCESS] Files saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()