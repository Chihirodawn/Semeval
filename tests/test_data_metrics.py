import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from task5_data import grouped_split, metrics, normalized, spearman


class DataMetricsTest(unittest.TestCase):
    def test_same_sentence_connects_different_targets(self):
        rows = [
            {"Compound": "A B", "Context": "shared sentence"},
            {"Compound": "C D", "Context": " Shared   sentence "},
            {"Compound": "A B", "Context": "another context"},
            {"Compound": "E F", "Context": "separate context"},
            {"Compound": "G H", "Context": "fourth context"},
        ]
        train, val, _ = grouped_split(rows, val_fraction=0.5)
        for column in ("Compound", "Context"):
            self.assertFalse({normalized(x[column]) for x in train} & {normalized(x[column]) for x in val})
        self.assertEqual(grouped_split(rows, val_fraction=0.5), grouped_split(rows, val_fraction=0.5))

    def test_ties_and_constants(self):
        self.assertAlmostEqual(spearman([1, 2, 2, 4], [2, 1, 1, 0]), -1)
        self.assertIsNone(spearman([1, 2, 3], [2, 2, 2]))

    def test_perfect_predictions_and_context_variation(self):
        rows = [{"Compound": t, "ModAvg": y, "HeadAvg": y} for t, y in [("a", 0), ("a", 1), ("b", 0), ("b", 4), ("c", 3)]]
        values = [[r["ModAvg"], r["HeadAvg"]] for r in rows]
        result = metrics(rows, values)
        self.assertAlmostEqual(result["mean_spearman"], 1)
        self.assertEqual(result["mean_rmse"], 0)
        self.assertEqual(result["ModAvg"]["singleton_targets"], 1)
        self.assertAlmostEqual(result["ModAvg"]["context_variance_spearman"], 1)


if __name__ == "__main__":
    unittest.main()
