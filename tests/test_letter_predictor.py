import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from letter_predictor import InvalidAttempt, validate_frames, verdict


class LetterAttemptTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(timesteps=18, window_seconds=2.94)
        self.frames = [dict(timestamp_ms=i * 100, jpeg='placeholder') for i in range(31)]

    def test_capture_duration(self):
        self.assertEqual(len(validate_frames(self.frames, self.config)), 31)
        with self.assertRaises(InvalidAttempt):
            validate_frames(self.frames[:18], self.config)

    def test_tracking_pause_and_non_monotonic_times(self):
        self.frames[15]['timestamp_ms'] = 2100
        with self.assertRaises(InvalidAttempt):
            validate_frames(self.frames, self.config)
        self.frames[15]['timestamp_ms'] = 1400
        with self.assertRaises(InvalidAttempt):
            validate_frames(self.frames, self.config)

    def test_invalid_payloads(self):
        for frames in [None, {}, [], [None] * 30]:
            with self.assertRaises(InvalidAttempt):
                validate_frames(frames, self.config)

    def test_outcomes(self):
        self.assertEqual(verdict([.995, .005], ['A', 'B'], 'A')['status'], 'match')
        self.assertEqual(verdict([.005, .995], ['A', 'B'], 'A')['status'], 'different_sign')
        self.assertEqual(verdict([.6, .4], ['A', 'B'], 'A')['status'], 'uncertain')

    def test_uncertain_results_do_not_identify_a_letter(self):
        for probabilities in ([.6, .4], [.849, .151], [float('nan'), .1]):
            result = verdict(probabilities, ['A', 'B'], 'A')
            self.assertEqual(result['status'], 'uncertain')
            self.assertIsNone(result['detected_letter'])

    def test_threshold_boundary_and_explicit_override(self):
        self.assertEqual(verdict([.85, .15], ['A', 'B'], 'A')['status'], 'match')
        self.assertEqual(verdict([.9, .1], ['A', 'B'], 'A', threshold=.8)['status'], 'match')


if __name__ == '__main__':
    unittest.main()
