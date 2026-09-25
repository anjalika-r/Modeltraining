import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from training_weights import signer_letter_weights


class WeightTests(unittest.TestCase):
    def test_equal_signer_mass_and_equal_letters_within_signer(self):
        m = pd.DataFrame([{'participant': p, 'class_id': c}
                          for p, c, n in [('a', 0, 3), ('a', 1, 7), ('b', 0, 2)]
                          for _ in range(n)], index=range(20, 32))
        m['weight'] = signer_letter_weights(m)
        np.testing.assert_allclose(m.groupby('participant').weight.sum(), [6, 6])
        np.testing.assert_allclose(m.groupby(['participant', 'class_id']).weight.sum(), [3, 3, 6])
        self.assertAlmostEqual(m.weight.mean(), 1)
        self.assertTrue((m.weight > 0).all())

    def test_invalid_metadata(self):
        for rows in [[], [{'participant': None, 'class_id': 0}]]:
            with self.assertRaises(ValueError):
                signer_letter_weights(pd.DataFrame(rows, columns=['participant', 'class_id']))


if __name__ == '__main__':
    unittest.main()
