import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from sign_features import normalize_body_hands, features_from_result, sample_window


class FeatureTests(unittest.TestCase):
    def raw_frame(self):
        raw = np.zeros(227, dtype=np.float32)
        raw[33:36] = [0.25, 0.5, -0.1]
        raw[36:39] = [0.75, 0.5, 0.1]
        raw[99:162] = np.tile([0.5, 0.75, 0.2], 21)
        raw[225] = 1
        return raw

    def test_missing_hand_stays_zero_and_flags_survive(self):
        result = normalize_body_hands(self.raw_frame())
        np.testing.assert_allclose(result[99:102], [0, 0.5, 0.4])
        np.testing.assert_array_equal(result[162:225], 0)
        np.testing.assert_array_equal(result[225:], [1, 0])

    def test_degenerate_shoulders_rejected(self):
        with self.assertRaises(ValueError):
            normalize_body_hands(np.zeros(227))

    def test_mediapipe_adapter_uses_image_landmarks(self):
        raw = self.raw_frame()
        def points(values):
            return [SimpleNamespace(x=x, y=y, z=z, visibility=1) for x, y, z in values.reshape(-1, 3)]
        result = SimpleNamespace(pose_landmarks=points(raw[:99]),
                                 left_hand_landmarks=points(raw[99:162]), right_hand_landmarks=[])
        np.testing.assert_array_equal(features_from_result(result), normalize_body_hands(raw))
        result.left_hand_landmarks = []
        self.assertIsNone(features_from_result(result))
        result.pose_landmarks = []
        self.assertIsNone(features_from_result(result))

    def test_resampling_matches_recording_metadata(self):
        import json
        import csv
        path = ROOT / 'data/letter_dataset/metadata.csv'
        if not path.exists():
            self.skipTest('Local participant data not present')
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                selected = sample_window(np.arange(int(row['standardizable_frames'])), int(row['target_frames']))
                np.testing.assert_array_equal(selected, json.loads(row['selected_source_indices']))

    def test_normalization_matches_all_saved_training_frames(self):
        path = ROOT / 'data/processed_18_frames'
        if not (path / 'sequences.npy').exists():
            self.skipTest('Local participant data not present')
        raw = np.load(path / 'sequences.npy', mmap_mode='r')
        expected = np.load(path / 'sequences_body_hands.npy', mmap_mode='r')
        for start in range(0, len(raw), 100):
            np.testing.assert_array_equal(normalize_body_hands(raw[start:start+100, :, 1404:]), expected[start:start+100])


if __name__ == '__main__':
    unittest.main()
