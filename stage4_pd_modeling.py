import json
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import (
    CAPPED_FEATURES,
    CAPPING_QUANTILE,
    CV_FOLDS,
    DELINQUENCY_FLAG,
    FEATURE_SETS,
    LOGISTIC_C_VALUES,
    MODEL_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RECORD_ID,
    SEPARATE_COUNT_FEATURES,
    STAGE4_OUTPUT_DIR,
    TARGET,
)
from features import engineer_features
from model_components import (
    IncomeLogTransformer,
    UpperQuantileCapper,
)


SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "neg_brier_score": "neg_brier_score",
    "neg_log_loss": "neg_log_loss",
}


def load_training_data():
    """Load training predictors without identifiers or outcomes."""
    df = pd.read_csv(PROCESSED_DIR / "train.csv")

    required = [RECORD_ID, TARGET] + SEPARATE_COUNT_FEATURES
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Training data is empty.")

    if df[RECORD_ID].isna().any() or df[RECORD_ID].duplicated().any():
        raise ValueError("Training record IDs must be present and unique.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Invalid training target values.")

    if df[TARGET].nunique() != 2:
        raise ValueError("Both target classes are required.")

    if df[TARGET].value_counts().min() < CV_FOLDS:
        raise ValueError("Insufficient records per class for CV.")

    X = df[SEPARATE_COUNT_FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            raise ValueError(f"{column} must be numeric.")

    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Predictors contain infinite values.")

    if (X < 0).any().any():
        raise ValueError("Predictors contain negative values.")

    if not X[DELINQUENCY_FLAG].isin([0, 1]).all():
        raise ValueError("The delinquency flag must contain 0 or 1.")

    if X.isna().all().any():
        raise ValueError("A training predictor is entirely missing.")

    return X, y


def build_pipeline(feature_names, estimator, scale):
    """Build a pipeline whose learned settings are fitted within CV."""
    capped_indices = tuple(
        index
        for index, feature in enumerate(feature_names)
        if feature in CAPPED_FEATURES
    )

    preprocessing_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        (
            "capper",
            UpperQuantileCapper(
                column_indices=capped_indices,
                quantile=CAPPING_QUANTILE,
            ),
        ),
        (
            "income_transform",
            IncomeLogTransformer(
                column_index=feature_names.index("monthly_inc"),
                enabled=False,
            ),
        ),
    ]

    if scale:
        preprocessing_steps.append(("scaler", StandardScaler()))

    selector = ColumnTransformer(
        transformers=[
            ("features", "passthrough", feature_names),
        ],
        remainder="drop",
    )

    return Pipeline([
        (
            "engineering",
            FunctionTransformer(engineer_features, validate=False),
        ),
        ("selection", selector),
        ("preprocessing", Pipeline(preprocessing_steps)),
        ("model", estimator),
    ])


def format_search_results(search, model_name, feature_set):
    """Export training and validation metrics for each configuration."""
    results = pd.DataFrame(search.cv_results_)

    output = pd.DataFrame({
        "model": model_name,
        "feature_set": feature_set,
        "parameters": [
            json.dumps(parameters, sort_keys=True)
            for parameters in results["params"]
        ],
        "mean_train_roc_auc": results["mean_train_roc_auc"],
        "mean_roc_auc": results["mean_test_roc_auc"],
        "std_roc_auc": results["std_test_roc_auc"],
        "mean_average_precision": results[
            "mean_test_average_precision"
        ],
        "mean_train_brier_score": -results[
            "mean_train_neg_brier_score"
        ],
        "mean_brier_score": -results["mean_test_neg_brier_score"],
        "mean_train_log_loss": -results["mean_train_neg_log_loss"],
        "mean_log_loss": -results["mean_test_neg_log_loss"],
        "mean_fit_seconds": results["mean_fit_time"],
    })

    output["train_validation_auc_gap"] = (
        output["mean_train_roc_auc"] - output["mean_roc_auc"]
    )

    return output


def save_training_diagnostics(X, y):
    """Save correlations and a descriptive debt-ratio profile."""
    X.corr(method="spearman").to_csv(
        STAGE4_OUTPUT_DIR / "training_spearman_correlations.csv"
    )

    debt_groups = pd.qcut(
        X["debt_ratio"],
        q=5,
        duplicates="drop",
    )

    profile_data = pd.DataFrame({
        "debt_ratio_group": (
            debt_groups.astype("string").fillna("Missing")
        ),
        TARGET: y,
    })

    debt_profile = (
        profile_data.groupby("debt_ratio_group", sort=False)[TARGET]
        .agg(
            records="size",
            events="sum",
            event_rate="mean",
        )
        .reset_index()
    )

    debt_profile.to_csv(
        STAGE4_OUTPUT_DIR / "training_debt_ratio_profile.csv",
        index=False,
    )


def save_logistic_coefficients(pipeline, feature_names):
    """Export standardised effects and the binary flag's 0-to-1 effect."""
    model = pipeline.named_steps["model"]
    preprocessing = pipeline.named_steps["preprocessing"]

    logged_income = bool(
        preprocessing.named_steps["income_transform"].enabled
    )

    coefficients = model.coef_[0]
    scales = preprocessing.named_steps["scaler"].scale_

    labels = [
        "log1p(monthly_inc)"
        if feature == "monthly_inc" and logged_income
        else feature
        for feature in feature_names
    ]

    table = pd.DataFrame({
        "feature": feature_names,
        "model_term": labels,
        "coefficient_per_standard_deviation": coefficients,
        "odds_ratio_per_standard_deviation": np.exp(coefficients),
    })

    table["odds_ratio_flag_0_to_1"] = np.nan
    flag_mask = table["feature"].eq(DELINQUENCY_FLAG).to_numpy()

    table.loc[flag_mask, "odds_ratio_flag_0_to_1"] = np.exp(
        coefficients[flag_mask] / scales[flag_mask]
    )

    table.sort_values(
        "coefficient_per_standard_deviation",
        ascending=False,
    ).to_csv(
        STAGE4_OUTPUT_DIR / "selected_logistic_coefficients.csv",
        index=False,
    )

    return logged_income


def main():
    print("[INFO] Stage 4: Training-set model comparison")

    X, y = load_training_data()

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = list(cv.split(X, y))

    logistic_grid = {
        "model__C": LOGISTIC_C_VALUES,
        "preprocessing__income_transform__enabled": [False, True],
    }

    experiments = [
        {
            "model_name": "naive_baseline",
            "feature_set": "separate_counts",
            "estimator": DummyClassifier(strategy="prior"),
            "scale": False,
            "grid": {},
        },
        {
            "model_name": "logistic_regression",
            "feature_set": "separate_counts",
            "estimator": LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            "scale": True,
            "grid": logistic_grid,
        },
        {
            "model_name": "logistic_regression",
            "feature_set": "weighted_composite",
            "estimator": LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            "scale": True,
            "grid": logistic_grid,
        },
        {
            "model_name": "gradient_boosting",
            "feature_set": "separate_counts",
            "estimator": HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_iter=150,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
            "scale": False,
            "grid": {"model__max_leaf_nodes": [7, 15]},
        },
    ]

    STAGE4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    best_rows = []
    fitted_candidates = []

    for experiment in experiments:
        model_name = experiment["model_name"]
        feature_set = experiment["feature_set"]
        feature_names = FEATURE_SETS[feature_set]

        print(f"[INFO] Comparing {model_name}: {feature_set}")

        pipeline = build_pipeline(
            feature_names,
            experiment["estimator"],
            experiment["scale"],
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=experiment["grid"],
            scoring=SCORING,
            refit="roc_auc",
            cv=splits,
            n_jobs=1,
            error_score="raise",
            return_train_score=True,
        )

        # Convergence warnings stop the run rather than being ignored.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            search.fit(X, y)

        results = format_search_results(
            search,
            model_name,
            feature_set,
        )
        all_results.append(results)
        best_rows.append(results.iloc[[search.best_index_]])

        fitted_candidates.append({
            "model": model_name,
            "feature_set": feature_set,
            "features": feature_names,
            "search": search,
            "model_file": f"{model_name}_{feature_set}.joblib",
        })

    logistic_candidates = [
        candidate
        for candidate in fitted_candidates
        if candidate["model"] == "logistic_regression"
    ]

    selected = max(
        logistic_candidates,
        key=lambda candidate: candidate["search"].best_score_,
    )

    selected_search = selected["search"]
    selected_pipeline = selected_search.best_estimator_
    fitted_model = selected_pipeline.named_steps["model"]

    iterations = int(fitted_model.n_iter_.max())
    below_iteration_limit = iterations < fitted_model.max_iter

    if not below_iteration_limit:
        raise RuntimeError(
            "The selected logistic model reached its iteration limit."
        )

    comparison = pd.concat(all_results, ignore_index=True)
    best_comparison = (
        pd.concat(best_rows, ignore_index=True)
        .sort_values("mean_roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    comparison.to_csv(
        STAGE4_OUTPUT_DIR / "cv_all_candidates.csv",
        index=False,
    )
    best_comparison.to_csv(
        STAGE4_OUTPUT_DIR / "cv_best_models.csv",
        index=False,
    )

    for candidate in fitted_candidates:
        joblib.dump(
            candidate["search"].best_estimator_,
            MODEL_DIR / candidate["model_file"],
        )

    selected_model_path = MODEL_DIR / "selected_logistic_pipeline.joblib"
    joblib.dump(selected_pipeline, selected_model_path)

    # Confirm that serialisation preserves predictions on training inputs.
    reloaded_pipeline = joblib.load(selected_model_path)
    probe = X.head(100)

    if not np.allclose(
        selected_pipeline.predict_proba(probe),
        reloaded_pipeline.predict_proba(probe),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("Reloaded model predictions do not match.")

    logged_income = save_logistic_coefficients(
        selected_pipeline,
        selected["features"],
    )
    save_training_diagnostics(X, y)

    metadata = {
        "selection_metric": "mean_cross_validated_roc_auc",
        "selected_logistic_feature_set": selected["feature_set"],
        "selected_logistic_features": selected["features"],
        "selected_logistic_parameters": selected_search.best_params_,
        "selected_logistic_cv_roc_auc": float(
            selected_search.best_score_
        ),
        "selected_logistic_intercept": float(
            fitted_model.intercept_[0]
        ),
        "income_log1p_selected": logged_income,
        "selected_logistic_iterations": iterations,
        "selected_logistic_max_iter": fitted_model.max_iter,
        "selected_logistic_below_iteration_limit": below_iteration_limit,
        "convergence_warnings_treated_as_errors": True,
        "model_reload_predictions_match": True,
        "training_records": len(X),
        "training_event_rate": float(y.mean()),
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "capping_quantile": CAPPING_QUANTILE,
        "capped_features": CAPPED_FEATURES,
        "imputation": "training_fold_median",
        "logistic_scaling": "training_fold_standard_scaler",
        "test_data_used": False,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "experiments": [
            {
                "model": candidate["model"],
                "feature_set": candidate["feature_set"],
                "best_parameters": candidate["search"].best_params_,
                "mean_cv_roc_auc": float(
                    candidate["search"].best_score_
                ),
                "model_file": candidate["model_file"],
            }
            for candidate in fitted_candidates
        ],
    }

    with (
        STAGE4_OUTPUT_DIR / "model_selection.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n[INFO] Best candidate from each comparison:")
    print(
        best_comparison[
            [
                "model",
                "feature_set",
                "mean_train_roc_auc",
                "mean_roc_auc",
                "train_validation_auc_gap",
                "mean_brier_score",
                "mean_log_loss",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        "\n[INFO] Selected logistic feature set: "
        f"{selected['feature_set']}"
    )
    print(
        "[INFO] Selected logistic parameters: "
        f"{selected_search.best_params_}"
    )
    print(
        f"[INFO] Selected logistic iterations: "
        f"{iterations}/{fitted_model.max_iter}"
    )
    print(f"[INFO] Selected logistic logged income: {logged_income}")
    print("[INFO] Saved model reload check passed")
    print("[INFO] Selected pipelines refitted on all training records")
    print("[INFO] CV scores are development estimates")
    print("[INFO] No test data was loaded")
    print(f"[SUCCESS] Results saved to: {STAGE4_OUTPUT_DIR}")
    print(f"[SUCCESS] Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()