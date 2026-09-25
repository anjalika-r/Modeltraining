import sys
from pathlib import Path
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dataset_split import split_selected_participants


class SplitTests(unittest.TestCase):
    def test_reproducible_disjoint_recordings_and_exclusion(self):
        rows = [dict(participant=p, class_id=c, source_file=f'{p}/{c}/{i}')
                for p in ['a', 'b', 'excluded'] for c in range(2) for i in range(20)]
        # Repeated rows must stay with their original recording.
        meta = pd.DataFrame(rows + rows[:10])
        train, val = split_selected_participants(meta, ['a', 'b'], 0.15, 42)
        again = split_selected_participants(meta, ['a', 'b'], 0.15, 42)
        self.assertTrue(train.equals(again[0]) and val.equals(again[1]))
        self.assertFalse((train & val).any())
        self.assertFalse((train | val)[meta.participant == 'excluded'].any())
        self.assertTrue((train | val)[meta.participant != 'excluded'].all())
        self.assertFalse(set(meta.loc[train, 'source_file']) & set(meta.loc[val, 'source_file']))
        self.assertEqual(len(meta.loc[val].drop_duplicates('source_file')), 12)

    def test_unknown_participant_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            split_selected_participants(pd.DataFrame({'participant': ['a']}), ['missing'], .15, 42)

    def test_preserves_previous_recordings(self):
        meta = pd.DataFrame([dict(participant=p, class_id=c, class_label=str(c), source_file=f'{p}/{c}/{i}')
                             for p in ['old', 'new'] for c in range(2) for i in range(20)])
        old = meta.loc[meta.participant == 'old'].copy()
        old['split'] = ['validation' if i % 5 == 0 else 'train' for i in range(len(old))]
        train, val = split_selected_participants(meta, ['old', 'new'], .15, 42, old)
        self.assertTrue(val.loc[old.index].equals(old.split.eq('validation')))
        self.assertTrue(train.loc[old.index].equals(old.split.eq('train')))
        self.assertEqual(val.loc[meta.participant == 'new'].sum(), 6)
        self.assertTrue((train | val).all())
        self.assertFalse((train & val).any())
        with self.assertRaisesRegex(ValueError, 'missing'):
            split_selected_participants(meta.iloc[1:], ['old', 'new'], .15, 42, old)
        old.loc[0, 'class_label'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'labels changed'):
            split_selected_participants(meta, ['old', 'new'], .15, 42, old)


if __name__ == '__main__':
    unittest.main()
