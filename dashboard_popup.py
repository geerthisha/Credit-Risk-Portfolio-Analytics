import json
import tkinter as tk

from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from config import (
    STAGE5_OUTPUT_DIR,
    STAGE6_OUTPUT_DIR,
    STAGE7_OUTPUT_DIR,
)


TITLE_FONT = ("Arial", 26, "bold")
BODY_FONT = ("Arial", 15)
BOLD_FONT = ("Arial", 15, "bold")
TABLE_FONT = ("Arial", 14)
TABLE_HEADING_FONT = ("Arial", 14, "bold")


def read_table(path, required_columns):
    """Load an output table and validate its required columns."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required output is missing:\n{path}"
        )

    table = pd.read_csv(path)
    missing = set(required_columns) - set(table.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{', '.join(sorted(missing))}"
        )

    if table.empty:
        raise ValueError(f"{path.name} contains no records.")

    return table


def read_json(path, required_keys):
    """Load a JSON object and validate its required fields."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required output is missing:\n{path}"
        )

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"{path.name} must contain a JSON object."
        )

    missing = set(required_keys) - set(data)

    if missing:
        raise ValueError(
            f"{path.name} is missing required fields: "
            f"{', '.join(sorted(missing))}."
        )

    return data


def validate_numeric(table, columns, filename):
    """Check that numerical output columns contain finite values."""
    for column in columns:
        try:
            values = pd.to_numeric(
                table[column],
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{filename}: {column} must be numeric."
            ) from error

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{filename}: {column} contains missing "
                "or non-finite values."
            )

        table[column] = values


def validate_score_audit(audit, borrowers):
    """Validate the lower-tail score count and share."""
    count = audit["scores_below_400"]
    share = audit["share_below_400"]

    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 <= count <= borrowers
    ):
        raise ValueError(
            "score_scaling_audit.json contains an invalid "
            "scores_below_400 value."
        )

    if (
        isinstance(share, bool)
        or not isinstance(share, (int, float))
        or not np.isfinite(share)
        or not 0 <= share <= 1
    ):
        raise ValueError(
            "score_scaling_audit.json contains an invalid "
            "share_below_400 value."
        )

    if not np.isclose(share, count / borrowers):
        raise ValueError(
            "The score-audit count and share below 400 disagree."
        )


def load_outputs():
    """Load and reconcile saved evaluation and portfolio outputs."""
    metrics = read_table(
        STAGE5_OUTPUT_DIR / "test_metrics.csv",
        [
            "model",
            "test_records",
            "roc_auc",
            "roc_auc_ci_lower",
            "roc_auc_ci_upper",
            "average_precision",
            "brier_score",
            "log_loss",
            "threshold",
        ],
    )

    bands = read_table(
        STAGE7_OUTPUT_DIR / "portfolio_band_summary.csv",
        [
            "risk_band",
            "borrowers",
            "observed_events",
            "borrower_share",
            "mean_predicted_probability",
            "observed_event_rate",
            "illustrative_loss",
            "loss_share",
        ],
    )

    totals = read_table(
        STAGE7_OUTPUT_DIR / "portfolio_totals.csv",
        [
            "borrowers",
            "observed_events",
            "currency",
            "assumed_ead_per_borrower",
            "assumed_lgd",
            "total_assumed_ead",
            "illustrative_loss",
            "illustrative_loss_rate",
        ],
    )

    paired = read_table(
        STAGE5_OUTPUT_DIR / "paired_auc_comparison.csv",
        [
            "test_auc_difference",
            "ci_lower",
            "ci_upper",
        ],
    )

    score_audit = read_json(
        STAGE6_OUTPUT_DIR / "score_scaling_audit.json",
        [
            "scores_below_400",
            "share_below_400",
        ],
    )

    evaluation_audit = read_json(
        STAGE5_OUTPUT_DIR / "evaluation_audit.json",
        [
            "confidence_level",
            "threshold",
            "models_refitted",
        ],
    )

    portfolio_audit = read_json(
        STAGE7_OUTPUT_DIR / "portfolio_audit.json",
        [
            "probability_edges",
            "prediction_horizon_years",
            "model_refitted",
        ],
    )

    validate_numeric(
        metrics,
        [
            "test_records",
            "roc_auc",
            "roc_auc_ci_lower",
            "roc_auc_ci_upper",
            "average_precision",
            "brier_score",
            "log_loss",
            "threshold",
        ],
        "test_metrics.csv",
    )

    validate_numeric(
        bands,
        [
            "borrowers",
            "observed_events",
            "borrower_share",
            "illustrative_loss",
        ],
        "portfolio_band_summary.csv",
    )

    validate_numeric(
        totals,
        [
            "borrowers",
            "observed_events",
            "assumed_ead_per_borrower",
            "assumed_lgd",
            "total_assumed_ead",
            "illustrative_loss",
            "illustrative_loss_rate",
        ],
        "portfolio_totals.csv",
    )

    validate_numeric(
        paired,
        ["test_auc_difference", "ci_lower", "ci_upper"],
        "paired_auc_comparison.csv",
    )

    if len(totals) != 1:
        raise ValueError("Expected one portfolio totals row.")

    if len(paired) != 1:
        raise ValueError("Expected one paired AUC comparison.")

    if metrics["model"].duplicated().any():
        raise ValueError("Model names must be unique.")

    required_models = {
        "naive_baseline",
        "logistic_regression",
        "gradient_boosting",
    }

    if not required_models.issubset(set(metrics["model"])):
        raise ValueError("Required model results are missing.")

    if bands["risk_band"].duplicated().any():
        raise ValueError("Risk-band labels must be unique.")

    total = totals.iloc[0]
    borrowers = int(total["borrowers"])

    if borrowers <= 0 or borrowers != total["borrowers"]:
        raise ValueError(
            "The total borrower count must be a positive integer."
        )

    for column in ["borrowers", "observed_events"]:
        values = bands[column]

        if (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(
                f"Band {column} must contain non-negative integers."
            )

    if (bands["observed_events"] > bands["borrowers"]).any():
        raise ValueError("Band events exceed borrower counts.")

    if bands["borrowers"].sum() != borrowers:
        raise ValueError("Portfolio borrower counts do not reconcile.")

    if bands["observed_events"].sum() != total["observed_events"]:
        raise ValueError("Portfolio event counts do not reconcile.")

    if not metrics["test_records"].eq(borrowers).all():
        raise ValueError(
            "Evaluation and portfolio sample sizes differ."
        )

    if not np.allclose(
        bands["borrower_share"],
        bands["borrowers"] / borrowers,
    ):
        raise ValueError("Band borrower shares do not reconcile.")

    if not np.isclose(
        bands["illustrative_loss"].sum(),
        total["illustrative_loss"],
    ):
        raise ValueError("Illustrative loss totals do not reconcile.")

    populated = bands["borrowers"] > 0

    validate_numeric(
        bands.loc[populated].copy(),
        [
            "mean_predicted_probability",
            "observed_event_rate",
        ],
        "populated portfolio bands",
    )

    if not np.allclose(
        bands.loc[populated, "observed_event_rate"],
        (
            bands.loc[populated, "observed_events"]
            / bands.loc[populated, "borrowers"]
        ),
    ):
        raise ValueError("Observed band rates do not reconcile.")

    if total["illustrative_loss"] > 0:
        validate_numeric(
            bands,
            ["loss_share"],
            "portfolio_band_summary.csv",
        )

        if not np.allclose(
            bands["loss_share"],
            bands["illustrative_loss"]
            / total["illustrative_loss"],
        ):
            raise ValueError("Band loss shares do not reconcile.")

    threshold = evaluation_audit["threshold"]

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 < threshold < 1
    ):
        raise ValueError("The evaluation threshold is invalid.")

    if not np.allclose(metrics["threshold"], threshold):
        raise ValueError("Evaluation thresholds do not agree.")

    confidence = evaluation_audit["confidence_level"]

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 < confidence < 1
    ):
        raise ValueError("The confidence level is invalid.")

    edges = np.asarray(
        portfolio_audit["probability_edges"],
        dtype=float,
    )

    if (
        edges.ndim != 1
        or len(edges) != len(bands) + 1
        or not np.isfinite(edges).all()
        or edges[0] != 0
        or edges[-1] != 1
        or not (np.diff(edges) > 0).all()
    ):
        raise ValueError(
            "Portfolio probability boundaries are invalid."
        )

    horizon = portfolio_audit["prediction_horizon_years"]

    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, (int, float))
        or not np.isfinite(horizon)
        or horizon <= 0
    ):
        raise ValueError("The prediction horizon is invalid.")

    if (
        evaluation_audit["models_refitted"] is not False
        or portfolio_audit["model_refitted"] is not False
    ):
        raise ValueError(
            "The supplied audits do not describe frozen-model reporting."
        )

    validate_score_audit(score_audit, borrowers)

    return {
        "metrics": metrics,
        "bands": bands,
        "totals": total,
        "paired": paired.iloc[0],
        "score_audit": score_audit,
        "evaluation_audit": evaluation_audit,
        "portfolio_audit": portfolio_audit,
    }


def add_text(parent, text, bold=False, title=False):
    """Add readable text that wraps to the available width."""
    font = (
        TITLE_FONT
        if title
        else BOLD_FONT if bold else BODY_FONT
    )

    label = ttk.Label(
        parent,
        text=text,
        wraplength=900,
        justify="left",
        font=font,
    )

    label.pack(
        fill="x",
        padx=20,
        pady=(14, 10) if title else 10,
    )

    def update_wrap(event):
        if event.widget is parent:
            label.configure(
                wraplength=max(200, event.width - 48)
            )

    parent.bind("<Configure>", update_wrap, add="+")
    return label


def add_table(parent, headings, rows):
    """Display a compact table with horizontal scrolling."""
    frame = ttk.Frame(parent)
    frame.pack(fill="x", padx=20, pady=12)

    table = ttk.Treeview(
        frame,
        columns=headings,
        show="headings",
        height=max(len(rows), 1),
    )

    for heading in headings:
        width = 240 if heading == "Model" else 185

        table.heading(heading, text=heading)
        table.column(
            heading,
            width=width,
            minwidth=140,
            anchor="center",
        )

    for row in rows:
        table.insert("", "end", values=row)

    horizontal = ttk.Scrollbar(
        frame,
        orient="horizontal",
        command=table.xview,
    )

    table.configure(xscrollcommand=horizontal.set)

    table.pack(fill="x")
    horizontal.pack(fill="x")


class ScrollableTab(ttk.Frame):
    """Provide vertical scrolling for each dashboard tab."""

    def __init__(self, parent):
        super().__init__(parent)

        self.canvas = tk.Canvas(
            self,
            highlightthickness=0,
            borderwidth=0,
        )

        scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
        )

        self.content = ttk.Frame(self.canvas)

        self.window_id = self.canvas.create_window(
            (0, 0),
            window=self.content,
            anchor="nw",
        )

        self.canvas.configure(
            yscrollcommand=scrollbar.set
        )

        self.canvas.pack(
            side="left",
            fill="both",
            expand=True,
        )
        scrollbar.pack(side="right", fill="y")

        self.content.bind(
            "<Configure>",
            self.update_scroll_region,
        )
        self.canvas.bind(
            "<Configure>",
            self.resize_content,
        )

    def update_scroll_region(self, event):
        self.canvas.configure(
            scrollregion=self.canvas.bbox("all")
        )

    def resize_content(self, event):
        self.canvas.itemconfigure(
            self.window_id,
            width=event.width,
        )


class RiskDashboard:
    """Present saved benchmark results without fitting models."""

    def __init__(self, root, outputs):
        self.root = root

        self.metrics = outputs["metrics"]
        self.bands = outputs["bands"]
        self.totals = outputs["totals"]
        self.paired = outputs["paired"]
        self.score_audit = outputs["score_audit"]
        self.evaluation_audit = outputs["evaluation_audit"]
        self.portfolio_audit = outputs["portfolio_audit"]

        self.chart_window = None
        self.chart_image = None

        root.title("Credit Risk Benchmark Analytics")

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        width = min(1250, screen_width - 80)
        height = min(850, screen_height - 100)

        root.geometry(f"{width}x{height}")
        root.minsize(
            min(950, width),
            min(650, height),
        )

        style = ttk.Style(root)

        style.configure(
            "TNotebook.Tab",
            font=BOLD_FONT,
            padding=(18, 10),
        )
        style.configure(
            "TButton",
            font=BODY_FONT,
            padding=(12, 10),
        )
        style.configure(
            "Treeview",
            font=TABLE_FONT,
            rowheight=36,
        )
        style.configure(
            "Treeview.Heading",
            font=TABLE_HEADING_FONT,
        )

        add_text(
            root,
            "Credit Risk Benchmark Analytics",
            title=True,
        )

        add_text(
            root,
            "How effectively can logistic regression predict two-year "
            "serious delinquency from borrower characteristics "
            "in this benchmark dataset?",
        )

        add_text(
            root,
            f"Held-out sample: {int(self.totals['borrowers']):,} "
            "borrowers. Results describe an approximately balanced "
            "benchmark. Models are frozen.",
        )

        notebook = ttk.Notebook(root)
        notebook.pack(
            fill="both",
            expand=True,
            padx=16,
            pady=12,
        )

        model_tab = ScrollableTab(notebook)
        portfolio_tab = ScrollableTab(notebook)
        charts_tab = ScrollableTab(notebook)

        notebook.add(
            model_tab,
            text="Model evaluation",
        )
        notebook.add(
            portfolio_tab,
            text="Risk concentration",
        )
        notebook.add(
            charts_tab,
            text="Charts and limitations",
        )

        self.build_model_tab(model_tab.content)
        self.build_portfolio_tab(portfolio_tab.content)
        self.build_charts_tab(charts_tab.content)

    def build_model_tab(self, parent):
        confidence = self.evaluation_audit["confidence_level"]
        threshold = self.evaluation_audit["threshold"]

        add_text(
            parent,
            "Held-out model performance",
            bold=True,
        )

        rows = []

        for row in self.metrics.itertuples(index=False):
            rows.append((
                row.model.replace("_", " ").title(),
                f"{row.roc_auc:.4f}",
                f"{row.roc_auc_ci_lower:.4f}–"
                f"{row.roc_auc_ci_upper:.4f}",
                f"{row.average_precision:.4f}",
                f"{row.brier_score:.4f}",
                f"{row.log_loss:.4f}",
            ))

        add_table(
            parent,
            [
                "Model",
                "ROC-AUC",
                f"{confidence:.0%} AUC interval",
                "Average precision",
                "Brier score",
                "Log loss",
            ],
            rows,
        )

        add_text(
            parent,
            "Higher ROC-AUC and average precision are better. "
            "Lower Brier score and log loss are better.",
        )

        add_text(
            parent,
            "Gradient boosting minus logistic ROC-AUC: "
            f"{self.paired['test_auc_difference']:.4f}; "
            f"paired {confidence:.0%} bootstrap interval "
            f"[{self.paired['ci_lower']:.4f}, "
            f"{self.paired['ci_upper']:.4f}].",
            bold=True,
        )

        add_text(
            parent,
            "Bootstrap intervals describe test-sample uncertainty "
            "conditional on the fitted models. They exclude training "
            "and model-selection uncertainty.",
        )

        add_text(
            parent,
            "The reporting threshold was fixed at "
            f"{threshold:.2f} before initial testing. "
            "It is not a validated lending decision policy.",
        )

    def build_portfolio_tab(self, parent):
        highest_band = self.bands.iloc[-1]
        highest_two = self.bands.tail(2)

        add_text(
            parent,
            "Risk and illustrative loss concentration",
            bold=True,
        )

        if self.totals["illustrative_loss"] > 0:
            add_text(
                parent,
                "The highest-risk band contains "
                f"{highest_band['borrower_share']:.1%} "
                "of borrowers and "
                f"{highest_band['loss_share']:.1%} "
                "of illustrative loss.",
                bold=True,
            )

            add_text(
                parent,
                "The highest two bands contain "
                f"{highest_two['borrower_share'].sum():.1%} "
                "of borrowers and "
                f"{highest_two['loss_share'].sum():.1%} "
                "of illustrative loss.",
            )
        else:
            add_text(
                parent,
                "Illustrative total loss is zero. "
                "Loss shares are undefined.",
                bold=True,
            )

        rows = []

        for row in self.bands.itertuples(index=False):
            populated = row.borrowers > 0

            rows.append((
                row.risk_band,
                f"{int(row.borrowers):,}",
                f"{row.borrower_share:.1%}",
                (
                    f"{row.mean_predicted_probability:.1%}"
                    if populated else "N/A"
                ),
                (
                    f"{row.observed_event_rate:.1%}"
                    if populated else "N/A"
                ),
                (
                    f"{row.loss_share:.1%}"
                    if self.totals["illustrative_loss"] > 0
                    else "N/A"
                ),
            ))

        add_table(
            parent,
            [
                "Probability band",
                "Borrowers",
                "Borrower share",
                "Predicted risk",
                "Observed rate",
                "Loss share",
            ],
            rows,
        )

        horizon = self.portfolio_audit[
            "prediction_horizon_years"
        ]

        add_text(
            parent,
            "Illustrative assumptions: "
            f"{self.totals['currency']} "
            f"{self.totals['assumed_ead_per_borrower']:,.0f} "
            "EAD per borrower; "
            f"LGD {self.totals['assumed_lgd']:.0%}; "
            f"{horizon:g}-year serious-delinquency probability "
            "used as a PD proxy.",
        )

        add_text(
            parent,
            "Illustrative total loss: "
            f"{self.totals['currency']} "
            f"{self.totals['illustrative_loss']:,.2f}. "
            "With constant EAD and LGD, loss share is proportional "
            "to the sum of predicted probabilities in each band.",
        )

        add_text(
            parent,
            "This is not an observed loss or an IFRS 9 provision. "
            "Probabilities and concentration patterns describe the "
            "benchmark sample; monetary totals are not estimates "
            "for a representative lending population.",
        )

    def build_charts_tab(self, parent):
        add_text(
            parent,
            "Charts",
            bold=True,
        )

        charts = [
            (
                "ROC curves",
                STAGE5_OUTPUT_DIR / "test_roc.png",
            ),
            (
                "Precision–recall curves",
                STAGE5_OUTPUT_DIR / "test_precision_recall.png",
            ),
            (
                "Calibration",
                STAGE5_OUTPUT_DIR / "test_calibration.png",
            ),
            (
                "Score distribution",
                STAGE6_OUTPUT_DIR / "test_score_distribution.png",
            ),
            (
                "Risk by band",
                STAGE7_OUTPUT_DIR / "risk_by_band.png",
            ),
            (
                "Borrowers by band",
                STAGE7_OUTPUT_DIR / "borrowers_by_band.png",
            ),
            (
                "Loss concentration",
                STAGE7_OUTPUT_DIR / "loss_concentration_by_band.png",
            ),
        ]

        for title, path in charts:
            ttk.Button(
                parent,
                text=title,
                command=lambda title=title, path=path: (
                    self.show_chart(title, path)
                ),
            ).pack(
                fill="x",
                padx=20,
                pady=6,
            )

        add_text(
            parent,
            "Interpretation and limitations",
            bold=True,
        )

        add_text(
            parent,
            "Sampling and earlier preprocessing are incompletely "
            "documented. Predictor timing is not fully specified, "
            "and the debt-ratio definition is ambiguous. "
            "Probabilities are not validated for a lending population.",
        )

        add_text(
            parent,
            "Scores below 400: "
            f"{self.score_audit['scores_below_400']:,} "
            f"({self.score_audit['share_below_400']:.2%}). "
            "Uncapped delinquency terms can extrapolate beyond "
            "training ranges.",
        )

        first = self.bands.iloc[0]

        add_text(
            parent,
            f"The lowest portfolio band ({first['risk_band']}) "
            f"contains {int(first['borrowers']):,} borrowers and "
            f"{int(first['observed_events']):,} "
            "serious-delinquency events. Interpret observed rates "
            "cautiously when event counts are small.",
        )

    def show_chart(self, title, path):
        try:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Chart is missing:\n{path}\n\n"
                    "Run the corresponding analysis stage."
                )

            with Image.open(path) as source:
                chart = source.copy()

            max_width = min(
                1200,
                self.root.winfo_screenwidth() - 100,
            )
            max_height = min(
                800,
                self.root.winfo_screenheight() - 150,
            )

            chart.thumbnail(
                (max_width, max_height),
                Image.Resampling.LANCZOS,
            )

            if self.chart_window is not None:
                if self.chart_window.winfo_exists():
                    self.chart_window.destroy()

            self.chart_window = tk.Toplevel(self.root)
            self.chart_window.title(title)

            self.chart_image = ImageTk.PhotoImage(
                chart,
                master=self.chart_window,
            )

            ttk.Label(
                self.chart_window,
                image=self.chart_image,
            ).pack(padx=16, pady=16)

        except Exception as error:
            messagebox.showerror(
                "Unable to display chart",
                str(error),
                parent=self.root,
            )


def main():
    root = tk.Tk()
    root.withdraw()

    try:
        outputs = load_outputs()
    except Exception as error:
        messagebox.showerror(
            "Unable to load project outputs",
            f"{error}\n\n"
            "Check the saved outputs from Stages 5–7.",
            parent=root,
        )
        root.destroy()
        return 1

    RiskDashboard(root, outputs)
    root.deiconify()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())