import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent

PREPARATION_AND_TRAINING = [
    "stage1_cleaning.py",
    "stage2_features.py",
    "stage3_woe_iv.py",
    "stage4_pd_modeling.py",
]

EVALUATION_AND_REPORTING = [
    "stage5_evaluation.py",
    "stage6_scorecard_scaling.py",
    "stage7_portfolio_analysis.py",
]


def run_stages(stages):
    """Run scripts sequentially and stop at the first failure."""
    for stage in stages:
        path = PROJECT_DIR / stage

        if not path.is_file():
            print(
                f"[ERROR] Missing script: {path}",
                file=sys.stderr,
            )
            return 1

    for stage in stages:
        print(f"\n[INFO] Running {stage}", flush=True)

        try:
            result = subprocess.run(
                [sys.executable, str(PROJECT_DIR / stage)],
                cwd=PROJECT_DIR,
                check=False,
            )
        except OSError as error:
            print(
                f"[ERROR] Could not start {stage}: {error}",
                file=sys.stderr,
            )
            return 1

        if result.returncode != 0:
            print(
                f"[ERROR] {stage} failed with exit code "
                f"{result.returncode}",
                file=sys.stderr,
            )
            return (
                result.returncode
                if result.returncode > 0
                else 1
            )

    print("\n[SUCCESS] All requested stages completed")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run the credit-risk analysis workflow."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Run Stages 1–7, including model fitting. "
            "Overwrites generated models and outputs."
        ),
    )
    args = parser.parse_args()

    stages = EVALUATION_AND_REPORTING.copy()

    if args.rebuild:
        stages = PREPARATION_AND_TRAINING + stages
        print(
            "[INFO] Rebuild requested: generated models and "
            "outputs will be overwritten.",
            flush=True,
        )
    else:
        print(
            "[INFO] Reporting mode: using existing fitted models.",
            flush=True,
        )

    return run_stages(stages)


if __name__ == "__main__":
    raise SystemExit(main())