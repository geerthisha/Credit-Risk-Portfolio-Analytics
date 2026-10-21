import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import PercentFormatter
from scipy.special import expit

from config import (
    ILLUSTRATIVE_CURRENCY,
    ILLUSTRATIVE_EAD,
    ILLUSTRATIVE_LGD,
    PORTFOLIO_BAND_LABELS,
    PORTFOLIO_PROBABILITY_EDGES,
    RECORD_ID,
    STAGE5_OUTPUT_DIR,
    STAGE6_OUTPUT_DIR,
    STAGE7_OUTPUT_DIR,
    TARGET,
)


PROBABILITY_COLUMN = "predicted_serious_delinquency"


def validate_settings():
    """Validate fixed bands and illustrative loss assumptions."""
    edges = np.asarray(
        PORTFOLIO_PROBABILITY_EDGES,
        dtype=float,
    )

    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("At least two probability boundaries are required.")

    if (
        not np.isfinite(edges).all()
        or edges[0] != 0
        or edges[-1] != 1
        or not (np.diff(edges) > 0).all()
    ):
        raise ValueError(
            "Band boundaries must increase strictly from zero to one."
        )

    if len(PORTFOLIO_BAND_LABELS) != len(edges) - 1:
        raise ValueError("Band labels do not match the boundaries.")

    if len(set(PORTFOLIO_BAND_LABELS)) != len(PORTFOLIO_BAND_LABELS):
        raise ValueError("Band labels must be unique.")

    if not np.isfinite(ILLUSTRATIVE_EAD) or ILLUSTRATIVE_EAD <= 0:
        raise ValueError("Illustrative EAD must be positive and finite.")

    if (
        not np.isfinite(ILLUSTRATIVE_LGD)
        or not 0 <= ILLUSTRATIVE_LGD <= 1
    ):
        raise ValueError("Illustrative LGD must be between zero and one.")


def load_and_validate_scores():
    """Reconcile scored records with the evaluated predictions."""
    scores = pd.read_csv(STAGE6_OUTPUT_DIR / "test_scores.csv")
    evaluated = pd.read_csv(
        STAGE5_OUTPUT_DIR / "test_predictions.csv"
    )

    required = [
        RECORD_ID,
        TARGET,
        PROBABILITY_COLUMN,
        "credit_score",
    ]
    missing = set(required) - set(scores.columns)

    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")

    if scores.empty:
        raise ValueError("The scored dataset is empty.")

    if (
        scores[RECORD_ID].isna().any()
        or scores[RECORD_ID].duplicated().any()
    ):
        raise ValueError("Record IDs must be present and unique.")

    if not scores[TARGET].isin([0, 1]).all():
        raise ValueError("Invalid target values.")

    probabilities = scores[PROBABILITY_COLUMN].to_numpy(dtype=float)

    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Invalid predicted probabilities.")

    if not np.isfinite(scores["credit_score"]).all():
        raise ValueError("Scores contain non-finite values.")

    comparison = scores[
        [RECORD_ID, TARGET, PROBABILITY_COLUMN]
    ].merge(
        evaluated[
            [RECORD_ID, TARGET, "logistic_regression_probability"]
        ],
        on=RECORD_ID,
        how="outer",
        suffixes=("_score", "_evaluated"),
        indicator=True,
        validate="one_to_one",
    )

    if not comparison["_merge"].eq("both").all():
        raise RuntimeError("Evaluation and score record IDs differ.")

    if not comparison[f"{TARGET}_score"].equals(
        comparison[f"{TARGET}_evaluated"]
    ):
        raise RuntimeError("Evaluation and score outcomes differ.")

    if not np.allclose(
        comparison[PROBABILITY_COLUMN],
        comparison["logistic_regression_probability"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("Probabilities differ from Stage 5.")

    return scores


def assign_bands(scores):
    """Assign fixed risk bands and calculate illustrative losses."""
    result = scores.copy()

    # The first band includes zero; later bands exclude their lower bound.
    result["risk_band"] = pd.cut(
        result[PROBABILITY_COLUMN],
        bins=PORTFOLIO_PROBABILITY_EDGES,
        labels=PORTFOLIO_BAND_LABELS,
        right=True,
        include_lowest=True,
    )

    if result["risk_band"].isna().any():
        raise RuntimeError("Some records were not assigned a band.")

    result["assumed_ead"] = ILLUSTRATIVE_EAD
    result["assumed_lgd"] = ILLUSTRATIVE_LGD

    result["illustrative_loss"] = (
        result[PROBABILITY_COLUMN]
        * result["assumed_ead"]
        * result["assumed_lgd"]
    )

    return result


def summarise_bands(records):
    """Summarise risk, borrower shares and illustrative loss shares."""
    summary = (
        records.groupby("risk_band", observed=False)
        .agg(
            borrowers=(RECORD_ID, "size"),
            observed_events=(TARGET, "sum"),
            mean_predicted_probability=(PROBABILITY_COLUMN, "mean"),
            observed_event_rate=(TARGET, "mean"),
            mean_score=("credit_score", "mean"),
            total_assumed_ead=("assumed_ead", "sum"),
            illustrative_loss=("illustrative_loss", "sum"),
        )
        .reset_index()
    )

    summary["borrower_share"] = (
        summary["borrowers"] / len(records)
    )

    total_loss = float(summary["illustrative_loss"].sum())

    if total_loss > 0:
        summary["loss_share"] = (
            summary["illustrative_loss"] / total_loss
        )
    else:
        summary["loss_share"] = np.nan

    summary["illustrative_loss_rate"] = (
        summary["illustrative_loss"]
        / summary["total_assumed_ead"].replace(0, np.nan)
    )

    if summary["borrowers"].sum() != len(records):
        raise RuntimeError("Band record counts do not reconcile.")

    if summary["observed_events"].sum() != records[TARGET].sum():
        raise RuntimeError("Band event counts do not reconcile.")

    if not np.isclose(summary["borrower_share"].sum(), 1.0):
        raise RuntimeError("Borrower shares do not sum to one.")

    if total_loss > 0:
        if not np.isclose(summary["loss_share"].sum(), 1.0):
            raise RuntimeError("Loss shares do not sum to one.")

    if not np.isclose(
        summary["total_assumed_ead"].sum(),
        records["assumed_ead"].sum(),
    ):
        raise RuntimeError("Exposure totals do not reconcile.")

    if not np.isclose(
        summary["illustrative_loss"].sum(),
        records["illustrative_loss"].sum(),
    ):
        raise RuntimeError("Illustrative loss totals do not reconcile.")

    return summary


def export_band_definitions(factor, offset):
    """Export score intervals corresponding to the probability bands."""
    rows = []

    for index, label in enumerate(PORTFOLIO_BAND_LABELS):
        lower_p = PORTFOLIO_PROBABILITY_EDGES[index]
        upper_p = PORTFOLIO_PROBABILITY_EDGES[index + 1]

        lower_score = (
            -np.inf
            if upper_p == 1
            else offset - factor * np.log(upper_p / (1 - upper_p))
        )
        upper_score = (
            np.inf
            if lower_p == 0
            else offset - factor * np.log(lower_p / (1 - lower_p))
        )

        rows.append({
            "risk_band": label,
            "probability_lower": lower_p,
            "probability_upper": upper_p,
            "probability_lower_inclusive": index == 0,
            "probability_upper_inclusive": True,
            "score_lower": lower_score,
            "score_upper": upper_score,
            "score_lower_inclusive": upper_p < 1,
            "score_upper_inclusive": False,
        })

    pd.DataFrame(rows).to_csv(
        STAGE7_OUTPUT_DIR / "risk_band_definitions.csv",
        index=False,
    )


def save_plots(summary):
    """Save risk, borrower-count and loss-concentration figures."""
    positions = np.arange(len(summary))
    labels = summary["risk_band"].astype(str)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        positions,
        summary["mean_predicted_probability"],
        marker="o",
        label="Mean predicted probability",
    )
    ax.plot(
        positions,
        summary["observed_event_rate"],
        marker="o",
        label="Observed serious-delinquency rate",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(
        xlabel="Fixed predicted-probability band",
        ylabel="Serious-delinquency probability / rate",
        title="Held-out risk bands: predicted versus observed",
        ylim=(0, 1.05),
    )
    ax.legend()

    fig.text(
        0.5,
        0.02,
        "Descriptive results for the balanced benchmark sample.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(
        STAGE7_OUTPUT_DIR / "risk_by_band.png",
        dpi=200,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        positions,
        summary["borrowers"],
        color="royalblue",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.bar_label(bars, padding=3)
    ax.set(
        xlabel="Fixed predicted-probability band",
        ylabel="Borrower count",
        title="Held-out borrower distribution by risk band",
    )
    ax.margins(y=0.15)

    fig.tight_layout()
    fig.savefig(
        STAGE7_OUTPUT_DIR / "borrowers_by_band.png",
        dpi=200,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.38

    ax.bar(
        positions - width / 2,
        summary["borrower_share"],
        width,
        label="Share of borrowers",
        color="royalblue",
    )
    ax.bar(
        positions + width / 2,
        summary["loss_share"],
        width,
        label="Share of illustrative loss",
        color="darkorange",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(
        xlabel="Fixed predicted-probability band",
        ylabel="Share of test-sample total",
        title="Borrower share versus illustrative loss share",
    )
    ax.legend()
    ax.margins(y=0.15)

    fig.text(
        0.5,
        0.02,
        "Constant assumed EAD and LGD; losses use benchmark probabilities.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(
        STAGE7_OUTPUT_DIR / "loss_concentration_by_band.png",
        dpi=200,
    )
    plt.close(fig)


def main():
    print("[INFO] Stage 7: Illustrative test-sample portfolio analysis")

    validate_settings()
    scores = load_and_validate_scores()

    with (
        STAGE6_OUTPUT_DIR / "score_scaling_audit.json"
    ).open(encoding="utf-8") as file:
        score_audit = json.load(file)

    factor = float(score_audit["factor"])
    offset = float(score_audit["offset"])

    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("The score scaling factor must be positive.")

    if not np.isfinite(offset):
        raise ValueError("The score scaling offset must be finite.")

    # This direction avoids taking logarithms of probabilities near 0 or 1.
    reconstructed_probabilities = expit(
        (offset - scores["credit_score"].to_numpy()) / factor
    )

    if not np.allclose(
        reconstructed_probabilities,
        scores[PROBABILITY_COLUMN],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("Scores and probabilities do not reconcile.")

    records = assign_bands(scores)
    summary = summarise_bands(records)

    total_ead = float(records["assumed_ead"].sum())
    total_loss = float(records["illustrative_loss"].sum())
    loss_rate = total_loss / total_ead

    observed_rates = summary.loc[
        summary["borrowers"] > 0,
        "observed_event_rate",
    ]

    observed_monotonic = bool(
        observed_rates.is_monotonic_increasing
    )

    highest_band = summary.iloc[-1]
    highest_two = summary.tail(2)

    concentration = pd.DataFrame([
        {
            "group": "highest_probability_band",
            "borrowers": int(highest_band["borrowers"]),
            "borrower_share": float(highest_band["borrower_share"]),
            "loss_share": float(highest_band["loss_share"]),
        },
        {
            "group": "highest_two_probability_bands",
            "borrowers": int(highest_two["borrowers"].sum()),
            "borrower_share": float(
                highest_two["borrower_share"].sum()
            ),
            "loss_share": (
                float(highest_two["loss_share"].sum())
                if total_loss > 0
                else np.nan
            ),
        },
    ])

    STAGE7_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records.to_csv(
        STAGE7_OUTPUT_DIR / "test_portfolio_records.csv",
        index=False,
    )
    summary.to_csv(
        STAGE7_OUTPUT_DIR / "portfolio_band_summary.csv",
        index=False,
    )
    concentration.to_csv(
        STAGE7_OUTPUT_DIR / "loss_concentration_summary.csv",
        index=False,
    )

    pd.DataFrame([{
        "borrowers": len(records),
        "observed_events": int(records[TARGET].sum()),
        "mean_predicted_probability": float(
            records[PROBABILITY_COLUMN].mean()
        ),
        "observed_event_rate": float(records[TARGET].mean()),
        "currency": ILLUSTRATIVE_CURRENCY,
        "assumed_ead_per_borrower": ILLUSTRATIVE_EAD,
        "assumed_lgd": ILLUSTRATIVE_LGD,
        "total_assumed_ead": total_ead,
        "illustrative_loss": total_loss,
        "illustrative_loss_rate": loss_rate,
    }]).to_csv(
        STAGE7_OUTPUT_DIR / "portfolio_totals.csv",
        index=False,
    )

    export_band_definitions(factor, offset)
    save_plots(summary)

    audit = {
        "sample": "held_out_test",
        "records": len(records),
        "band_method": "fixed probability boundaries",
        "probability_edges": PORTFOLIO_PROBABILITY_EDGES,
        "boundary_selection": (
            "Descriptive boundaries chosen independently of test outcomes."
        ),
        "currency": ILLUSTRATIVE_CURRENCY,
        "assumed_ead_per_borrower": ILLUSTRATIVE_EAD,
        "assumed_lgd": ILLUSTRATIVE_LGD,
        "prediction_horizon_years": 2,
        "loss_formula": (
            "sum(predicted serious-delinquency probability "
            "* assumed EAD * assumed LGD)"
        ),
        "loss_concentration_interpretation": (
            "With constant EAD and LGD, loss shares are proportional "
            "to each band's sum of predicted event probabilities."
        ),
        "loss_shares_defined": total_loss > 0,
        "serious_delinquency_used_as_pd_proxy": True,
        "discounting_applied": False,
        "probabilities_match_stage5": True,
        "scores_match_probabilities": True,
        "band_totals_reconcile": True,
        "borrower_shares_sum_to_one": True,
        "loss_shares_sum_to_one": (
            True if total_loss > 0 else None
        ),
        "observed_event_rates_monotonic_increasing": observed_monotonic,
        "model_refitted": False,
        "model_sha256_from_stage6": score_audit["model_sha256"],
        "limitations": [
            "The sample has approximately equal event and non-event counts.",
            "Probabilities have not been recalibrated to a lending population.",
            "EAD and LGD are assumptions, not observed borrower quantities.",
            "Serious delinquency is a proxy for default in this illustration.",
            "Monetary losses are not representative portfolio estimates.",
            "No observed monetary losses are available for validation.",
            "This calculation does not establish IFRS 9 compliance.",
            "Band and concentration patterns describe this benchmark only.",
        ],
    }

    with (
        STAGE7_OUTPUT_DIR / "portfolio_audit.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)

    print("\n[INFO] Held-out risk-band summary:")
    print(
        summary[
            [
                "risk_band",
                "borrowers",
                "borrower_share",
                "mean_predicted_probability",
                "observed_event_rate",
                "loss_share",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n[INFO] Illustrative loss concentration:")

    for row in concentration.itertuples(index=False):
        label = row.group.replace("_", " ").capitalize()

        if total_loss > 0:
            print(
                f"[INFO] {label}: "
                f"{row.borrower_share:.2%} of borrowers; "
                f"{row.loss_share:.2%} of illustrative loss"
            )
        else:
            print(
                f"[INFO] {label}: "
                f"{row.borrower_share:.2%} of borrowers; "
                "loss share undefined because total loss is zero"
            )

    print(
        f"\n[INFO] Assumptions: "
        f"{ILLUSTRATIVE_CURRENCY} {ILLUSTRATIVE_EAD:,.0f} EAD "
        f"per borrower; LGD {ILLUSTRATIVE_LGD:.0%}"
    )
    print(
        f"[INFO] Total assumed EAD: "
        f"{ILLUSTRATIVE_CURRENCY} {total_ead:,.2f}"
    )
    print(
        f"[INFO] Illustrative two-year loss: "
        f"{ILLUSTRATIVE_CURRENCY} {total_loss:,.2f}"
    )
    print(f"[INFO] Illustrative loss rate: {loss_rate:.2%}")
    print(
        "[INFO] Observed rates increase monotonically across bands: "
        f"{observed_monotonic}"
    )
    print("[INFO] All band totals and defined shares reconcile")
    print("[INFO] No model fitting or probability adjustment performed")
    print("[INFO] Monetary results are illustrative benchmark calculations")
    print(f"[SUCCESS] Files saved to: {STAGE7_OUTPUT_DIR}")


if __name__ == "__main__":
    main()