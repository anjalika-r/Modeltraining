import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dataset_split import split_from_manifest


class ExplicitSplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({'participant': ['a'] * 3,
                                  'source_file': ['one', 'two', 'three'],
                                  'class_label': ['A'] * 3})
        self.manifest = self.rows.assign(split=['train', 'validation', 'test'])

    def test_reordered_manifest_aligns_by_recording(self):
        masks = split_from_manifest(self.rows, self.manifest.iloc[::-1])
        self.assertEqual([m.tolist() for m in masks],
                         [[True, False, False], [False, True, False], [False, False, True]])

    def test_incomplete_duplicate_or_relabeled_manifest_rejected(self):
        changed = self.manifest.copy()
        changed.loc[0, 'class_label'] = 'B'
        duplicate = pd.concat([self.manifest.iloc[:2], self.manifest.iloc[:1]])
        for manifest in [self.manifest.iloc[:2], duplicate, changed]:
            with self.assertRaises(ValueError):
                split_from_manifest(self.rows, manifest)

    def test_missing_test_or_invalid_assignment_rejected(self):
        for assignment in ['train', 'other', None]:
            changed = self.manifest.copy()
            changed.loc[2, 'split'] = assignment
            with self.assertRaises(ValueError):
                split_from_manifest(self.rows, changed)
