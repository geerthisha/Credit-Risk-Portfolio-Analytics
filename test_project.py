import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from scipy.special import expit

from config import (
    MODEL_DIR,
    PROCESSED_DIR,
    PROJECT_DIR,
    RECORD_ID,
    SEPARATE_COUNT_FEATURES,
    STAGE5_OUTPUT_DIR,
    STAGE6_OUTPUT_DIR,
    TARGET,
)
from features import engineer_features
from model_components import UpperQuantileCapper


class TransformationTests(unittest.TestCase):
    def test_cap_is_learned_from_training_values(self):
        training = np.array([
            [0.0, 10.0],
            [2.0, 20.0],
            [4.0, 30.0],
        ])

        transformer = UpperQuantileCapper(
            column_indices=(0,),
            quantile=0.5,
        )
        transformer.fit(training)

        transformed = transformer.transform(
            np.array([[1000.0, 999.0]])
        )

        np.testing.assert_allclose(
            transformed,
            [[2.0, 999.0]],
        )
        self.assertEqual(transformer.caps_[0], 2.0)

    def test_weighted_total_preserves_missing_components(self):
        data = pd.DataFrame({
            "late_30_59": [1.0, np.nan],
            "late_60_89": [2.0, 1.0],
            "late_90": [3.0, 0.0],
        })
        original = data.copy(deep=True)

        result = engineer_features(data)

        self.assertEqual(result.loc[0, "total_delinquency"], 14.0)
        self.assertTrue(
            pd.isna(result.loc[1, "total_delinquency"])
        )
        pd.testing.assert_frame_equal(data, original)


class FrozenOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(PROCESSED_DIR / "train.csv")
        cls.test = pd.read_csv(PROCESSED_DIR / "test.csv")
        cls.predictions = pd.read_csv(
            STAGE5_OUTPUT_DIR / "test_predictions.csv"
        )
        cls.scores = pd.read_csv(
            STAGE6_OUTPUT_DIR / "test_scores.csv"
        )
        cls.contributions = pd.read_csv(
            STAGE6_OUTPUT_DIR / "test_score_contributions.csv"
        )

        with (
            STAGE6_OUTPUT_DIR / "score_scaling_audit.json"
        ).open(encoding="utf-8") as file:
            cls.score_audit = json.load(file)

    def test_split_separation(self):
        self.assertTrue(self.train[RECORD_ID].is_unique)
        self.assertTrue(self.test[RECORD_ID].is_unique)

        self.assertFalse(
            set(self.train[RECORD_ID])
            & set(self.test[RECORD_ID])
        )

        duplicate_columns = SEPARATE_COUNT_FEATURES + [TARGET]

        combined = pd.concat(
            [
                self.train[duplicate_columns],
                self.test[duplicate_columns],
            ],
            ignore_index=True,
        )

        self.assertFalse(combined.duplicated().any())

    def test_probability_bounds_and_record_coverage(self):
        self.assertTrue(self.predictions[RECORD_ID].is_unique)
        self.assertEqual(
            set(self.predictions[RECORD_ID]),
            set(self.test[RECORD_ID]),
        )

        probability_columns = [
            "naive_baseline_probability",
            "logistic_regression_probability",
            "gradient_boosting_probability",
        ]

        values = self.predictions[
            probability_columns
        ].to_numpy()

        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_scores_match_probabilities_and_contributions(self):
        self.assertTrue(self.scores[RECORD_ID].is_unique)
        self.assertTrue(self.contributions[RECORD_ID].is_unique)

        self.assertEqual(
            set(self.scores[RECORD_ID]),
            set(self.contributions[RECORD_ID]),
        )

        joined = self.scores.merge(
            self.contributions,
            on=RECORD_ID,
            validate="one_to_one",
        )

        factor = self.score_audit["factor"]
        offset = self.score_audit["offset"]

        recovered = expit(
            (offset - joined["credit_score"]) / factor
        )

        np.testing.assert_allclose(
            recovered,
            joined["predicted_serious_delinquency"],
            rtol=1e-10,
            atol=1e-12,
        )

        point_columns = [
            column
            for column in self.contributions.columns
            if column.endswith("_points")
            and column != "base_points"
        ]

        self.assertGreater(len(point_columns), 0)

        reconstructed = (
            joined["base_points"]
            + joined[point_columns].sum(axis=1)
        )

        np.testing.assert_allclose(
            reconstructed,
            joined["credit_score"],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_fresh_process_model_loading_matches_evaluation(self):
        model_path = MODEL_DIR / "selected_logistic_pipeline.joblib"
        model = joblib.load(model_path)

        X = self.test[SEPARATE_COUNT_FEATURES]
        current = model.predict_proba(X)[:, 1]

        expected = (
            self.test[[RECORD_ID]]
            .merge(
                self.predictions[
                    [RECORD_ID, "logistic_regression_probability"]
                ],
                on=RECORD_ID,
                validate="one_to_one",
            )["logistic_regression_probability"]
            .to_numpy()
        )

        np.testing.assert_allclose(
            current,
            expected,
            rtol=1e-10,
            atol=1e-12,
        )

        child_code = """
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])

import joblib
import numpy as np
import pandas as pd

from config import (
    MODEL_DIR,
    PROCESSED_DIR,
    SEPARATE_COUNT_FEATURES,
)

data = pd.read_csv(PROCESSED_DIR / "test.csv")
model = joblib.load(
    MODEL_DIR / "selected_logistic_pipeline.joblib"
)

np.save(
    sys.argv[2],
    model.predict_proba(data[SEPARATE_COUNT_FEATURES])[:, 1],
)
"""

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "predictions.npy"

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    child_code,
                    str(PROJECT_DIR),
                    str(output_path),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stderr,
            )

            np.testing.assert_allclose(
                np.load(output_path),
                expected,
                rtol=1e-10,
                atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)