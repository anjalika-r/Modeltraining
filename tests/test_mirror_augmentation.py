import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sign_features import mirror_body_hands, normalize_body_hands, model_features


class MirrorTests(unittest.TestCase):
    def test_reflection_matches_raw_image_reflection_and_side_swap(self):
        raw = np.random.default_rng(42).random((2, 4, 227), dtype=np.float32)
        raw[..., 225:] = 1
        raw[..., 33:36] = [0.2, 0.4, 0.1]
        raw[..., 36:39] = [0.8, 0.4, 0.2]
        expected_raw = raw.copy()
        pose = expected_raw[..., :99].reshape(2, 4, 33, 3)
        pairs = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)]
        pairs += [(i, i + 1) for i in range(11, 33, 2)]
        for left, right in pairs:
            pose[..., [left, right], :] = pose[..., [right, left], :]
        expected_raw[..., 99:162] = raw[..., 162:225]
        expected_raw[..., 162:225] = raw[..., 99:162]
        expected_raw[..., :225:3] = 1 - expected_raw[..., :225:3]
        normalized = normalize_body_hands(raw)
        before = normalized.copy()
        actual = mirror_body_hands(normalized)
        np.testing.assert_allclose(actual, normalize_body_hands(expected_raw), atol=2e-7)
        np.testing.assert_array_equal(normalized, before)
        np.testing.assert_array_equal(mirror_body_hands(actual), normalized)

    def test_missing_hand_and_derived_geometry_move_to_opposite_side(self):
        frame = np.zeros(227, dtype=np.float32)
        frame[99:162] = np.random.default_rng(3).random(63)
        frame[225] = 1
        mirrored = mirror_body_hands(frame)
        np.testing.assert_array_equal(mirrored[99:162], 0)
        np.testing.assert_array_equal(mirrored[225:], [0, 1])
        np.testing.assert_array_equal(mirrored[162:225:3], -frame[99:162:3])
        original_features = model_features(frame)
        features = model_features(mirrored)
        # Joint angles and extension ratios are reflection invariant, but change slots.
        np.testing.assert_allclose(features[373:393], original_features[290:310])
        np.testing.assert_array_equal(features[227:310], 0)
        np.testing.assert_array_equal(features[393:495], 0)
        np.testing.assert_array_equal(features[495:], [0, 1])

    def test_invalid_input_rejected(self):
        for value in [np.zeros(497), np.full(227, np.nan), np.array(1)]:
            with self.assertRaises(ValueError):
                mirror_body_hands(value)


if __name__ == '__main__':
    unittest.main()
