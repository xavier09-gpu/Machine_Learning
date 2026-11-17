import unittest
from pathlib import Path

import numpy as np

from src.run_experiments import ROOT, hog_features, load_dataset


class DatasetTests(unittest.TestCase):
    def test_official_splits_have_expected_shape_and_labels(self) -> None:
        x_train, y_train = load_dataset(ROOT / "data" / "raw" / "optdigits.tra")
        x_test, y_test = load_dataset(ROOT / "data" / "raw" / "optdigits.tes")

        self.assertEqual(x_train.shape, (3823, 64))
        self.assertEqual(y_train.shape, (3823,))
        self.assertEqual(x_test.shape, (1797, 64))
        self.assertEqual(y_test.shape, (1797,))
        self.assertTrue(np.isin(y_train, np.arange(10)).all())
        self.assertTrue(np.isin(y_test, np.arange(10)).all())

    def test_hog_preserves_sample_count_and_is_finite(self) -> None:
        x_train, _ = load_dataset(ROOT / "data" / "raw" / "optdigits.tra")
        descriptors = hog_features(x_train[:5])

        self.assertEqual(descriptors.shape, (5, 324))
        self.assertTrue(np.isfinite(descriptors).all())


if __name__ == "__main__":
    unittest.main()
