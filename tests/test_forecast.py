import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from forecast import prepare, forecast_frame, score


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "history.csv"
        self.data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=64, freq="W-MON"),
                "ticket": np.arange(64) + 50,
                "traffic": np.arange(64) + 100,
                "promotion": np.arange(64) % 2,
            }
        )
        self.data.to_csv(self.path, index=False)

    def test_holdout_excludes_future_targets_and_past_covariates(self):
        arrays, dates, train, actual = prepare(
            self.path, ["ticket"], ["traffic"], ["promotion"], 8, "W-MON", holdout=True
        )
        self.assertEqual(arrays["context"].shape, (1, 56))
        self.assertEqual(arrays["past_only_covariates"].shape, (1, 56))
        self.assertEqual(arrays["past_future_covariates"].shape, (1, 64))
        self.assertEqual(arrays["context"][0, -1], 105)
        self.assertEqual(dates[0], actual.date.iloc[0])
        self.assertLess(train.date.iloc[-1], actual.date.iloc[0])

    def test_gap_rejected(self):
        self.data.drop(index=20).to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            prepare(self.path, ["ticket"], [], [], 8, "W-MON")

    def test_missing_value_rejected(self):
        self.data.loc[0, "ticket"] = np.nan
        self.data.to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            prepare(self.path, ["ticket"], [], [], 8, "W-MON")

    def test_misaligned_future_rejected(self):
        future = Path(self.temp.name) / "future.csv"
        self.data.tail(8)[["date", "promotion"]].to_csv(future, index=False)
        with self.assertRaises(ValueError):
            prepare(self.path, ["ticket"], [], ["promotion"], 8, "W-MON", future)

    def test_output_mapping_and_metrics(self):
        arrays, dates, train, actual = prepare(
            self.path, ["ticket"], [], [], 8, "W-MON", holdout=True
        )
        values = actual[["ticket"]].to_numpy().T
        output = SimpleNamespace(
            forecast=values, quantiles=np.repeat(values[..., None], 9, axis=-1)
        )
        result = forecast_frame(output, dates, ["ticket"])
        self.assertEqual(len(result), 8)
        self.assertEqual(result.p50.tolist(), result.forecast.tolist())
        self.assertEqual(score(result, actual, train, ["ticket"])["ticket"]["mae"], 0)


if __name__ == "__main__":
    unittest.main()
