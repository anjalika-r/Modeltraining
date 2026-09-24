import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sign_features import model_features, SCHEMA


class EngineeredTests(unittest.TestCase):
    def frame(self):
        x = np.zeros(227, dtype=np.float32)
        h = np.zeros((21, 3), dtype=np.float32)
        for finger in range(5):
            for joint in range(4):
                h[1 + finger * 4 + joint] = [finger * .2 - .4, .3 + joint * .2, 0]
        x[99:162] = h.ravel()
        x[162:225] = (h + [1, 0, 0]).ravel()
        x[225:] = 1
        return x

    def test_geometry_and_frame_batch_parity(self):
        x = self.frame()
        f = model_features(x)
        self.assertEqual(f.shape, (497,))
        np.testing.assert_allclose(f[305:310], 1)  # Straight fingers
        np.testing.assert_allclose(f[393:493].reshape(10, 10).diagonal(), 1)
        np.testing.assert_allclose(f[493:495], [1, 0])
        np.testing.assert_array_equal(f[-2:], [1, 1])
        np.testing.assert_array_equal(model_features(np.stack([x, x])), np.stack([f, f]))
        np.testing.assert_array_equal(model_features(x, SCHEMA), x)

    def test_local_shape_ignores_translation_and_uniform_scale(self):
        x = self.frame()
        y = x.copy()
        y[99:225] = (y[99:225].reshape(42, 3) * 2 + [3, 4, 5]).ravel()
        np.testing.assert_allclose(model_features(x)[227:393], model_features(y)[227:393], atol=2e-4)

    def test_missing_or_collapsed_hand_masks_relations(self):
        for missing in (True, False):
            x = self.frame()
            if missing:
                x[226] = 0  # Even stale coordinates must be ignored.
            else:
                x[162:225] = 3
            f = model_features(x)
            self.assertTrue(np.isfinite(f).all())
            np.testing.assert_array_equal(f[162:225], 0)
            np.testing.assert_array_equal(f[310:495], 0)
            np.testing.assert_array_equal(f[-2:], [1, 0])

    def test_cross_relations_do_not_use_independent_hand_depths(self):
        x = self.frame()
        y = x.copy()
        y[164:225:3] += 100
        np.testing.assert_array_equal(model_features(x)[393:495], model_features(y)[393:495])


if __name__ == '__main__':
    unittest.main()
