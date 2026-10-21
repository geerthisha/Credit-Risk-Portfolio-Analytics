from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_FILE = (
    PROJECT_DIR / "data" / "raw" / "Credit Risk Benchmark Dataset.csv"
)
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

TARGET = "dlq_2yrs"
RECORD_ID = "record_id"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MINIMUM_AGE = 18

RAW_PREDICTORS = [
    "rev_util",
    "age",
    "late_30_59",
    "debt_ratio",
    "monthly_inc",
    "open_credit",
    "late_90",
    "real_estate",
    "late_60_89",
    "dependents",
]

DELINQUENCY_COLUMNS = [
    "late_30_59",
    "late_60_89",
    "late_90",
]

COUNT_COLUMNS = [
    "late_30_59",
    "open_credit",
    "late_90",
    "real_estate",
    "late_60_89",
    "dependents",
]

SUSPICIOUS_DELINQUENCY_VALUES = [96, 98]
DELINQUENCY_FLAG = "delinquency_code_flag"

ENGINEERED_FEATURES = ["total_delinquency"]

SEPARATE_COUNT_FEATURES = RAW_PREDICTORS + [DELINQUENCY_FLAG]

COMPOSITE_FEATURES = [
    feature
    for feature in RAW_PREDICTORS
    if feature not in DELINQUENCY_COLUMNS
] + [DELINQUENCY_FLAG, "total_delinquency"]

CANDIDATE_MODEL_FEATURES = (
    SEPARATE_COUNT_FEATURES + ENGINEERED_FEATURES
)

FEATURE_SETS = {
    "separate_counts": SEPARATE_COUNT_FEATURES,
    "weighted_composite": COMPOSITE_FEATURES,
}

STAGE4_OUTPUT_DIR = PROJECT_DIR / "outputs" / "stage4"
MODEL_DIR = PROJECT_DIR / "models"

CV_FOLDS = 5
CAPPING_QUANTILE = 0.99
CAPPED_FEATURES = ["rev_util", "debt_ratio"]

LOGISTIC_C_VALUES = [0.1, 1.0, 10.0]

STAGE5_OUTPUT_DIR = PROJECT_DIR / "outputs" / "stage5"

EVALUATION_THRESHOLD = 0.50
BOOTSTRAP_REPETITIONS = 1000
CALIBRATION_BINS = 10

STAGE6_OUTPUT_DIR = PROJECT_DIR / "outputs" / "stage6"

SCORE_REFERENCE_POINTS = 600.0
SCORE_REFERENCE_NON_EVENT_ODDS = 1.0
SCORE_POINTS_TO_DOUBLE_ODDS = 20.0

STAGE7_OUTPUT_DIR = PROJECT_DIR / "outputs" / "stage7"

PORTFOLIO_PROBABILITY_EDGES = [
    0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0
]

PORTFOLIO_BAND_LABELS = [
    "0-10%",
    "10-20%",
    "20-40%",
    "40-60%",
    "60-80%",
    "80-100%",
]

ILLUSTRATIVE_EAD = 10_000.0
ILLUSTRATIVE_LGD = 0.45
ILLUSTRATIVE_CURRENCY = "GBP"