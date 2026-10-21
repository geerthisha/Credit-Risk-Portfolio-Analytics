from config import DELINQUENCY_COLUMNS


def engineer_features(df):
    """Add a severity-weighted measure of recorded delinquency."""
    missing = set(DELINQUENCY_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing delinquency columns: {sorted(missing)}"
        )

    result = df.copy()

    # Weights represent a chosen severity convention.
    # Missing components propagate to the weighted total.
    result["total_delinquency"] = (
        result["late_30_59"]
        + 2 * result["late_60_89"]
        + 3 * result["late_90"]
    )

    return result