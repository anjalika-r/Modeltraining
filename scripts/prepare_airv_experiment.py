"""Prepare AIRV without R and freeze recording splits before mirroring."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sign_features import mirror_body_hands


def main():
    source = Path('data/AIRV_dataset')
    output = Path('data/AIRV_no_r')
    if output.exists():
        raise SystemExit('Output already exists; preserve the frozen dataset.')
    metadata = pd.read_csv(source / 'metadata.csv')
    keep = metadata.class_label.ne('R').to_numpy()
    x = np.load(source / 'sequences_body_hands.npy')[keep]
    rows = metadata.loc[keep].copy().reset_index(drop=True)
    rows.participant = rows.participant.replace({'isaac copy': 'isaac'})
    names = sorted(rows.class_label.unique())
    mapping = {name: i for i, name in enumerate(names)}
    rows.class_id = rows.class_label.map(mapping)
    rows['sequence_sha256'] = [hashlib.sha256(a.tobytes()).hexdigest() for a in x]
    previous = pd.read_csv('models/letters_three_signers_mirrored/split_manifest.csv')
    previous = previous.loc[previous.class_label.ne('R')]
    assignments = previous.set_index(['participant', 'source_file']).split
    rows['split'] = pd.MultiIndex.from_frame(rows[['participant', 'source_file']]).map(assignments)
    new = rows.loc[rows.split.isna()]
    if set(new.participant) != {'isaac'}:
        raise ValueError('Unexpected new participants')
    _, validation = train_test_split(new.index, test_size=.15, random_state=42,
                                    stratify=new.class_label)
    rows.loc[new.index, 'split'] = 'train'
    rows.loc[validation, 'split'] = 'validation'
    # Keep identical tensors together; reserve test clips from training candidates.
    candidates = rows.loc[rows.split.eq('train')].drop_duplicates('sequence_sha256')
    strata = candidates.participant + ':' + candidates.class_label
    eligible = strata.map(strata.value_counts()).ge(2)
    _, test = train_test_split(candidates.loc[eligible].index, test_size=.15,
                              random_state=43, stratify=strata.loc[eligible])
    rows.loc[rows.sequence_sha256.isin(rows.loc[test, 'sequence_sha256']), 'split'] = 'test'
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if set(rows.loc[rows.split.eq(a), 'sequence_sha256']) & set(rows.loc[rows.split.eq(b), 'sequence_sha256']):
            raise ValueError('Duplicate tensors cross split boundaries')
    mirrored_hashes = {hashlib.sha256(a.tobytes()).hexdigest()
                       for a in mirror_body_hands(x[rows.split.eq('train')])}
    if mirrored_hashes & set(rows.loc[rows.split.ne('train'), 'sequence_sha256']):
        raise ValueError('Mirrored training tensors overlap evaluation')
    np.testing.assert_array_equal(mirror_body_hands(mirror_body_hands(x)), x)
    output.mkdir()
    np.save(output / 'sequences_body_hands.npy', x)
    np.save(output / 'labels.npy', rows.class_id.to_numpy(dtype=np.int64))
    np.save(output / 'labels_text.npy', rows.class_label.to_numpy(dtype=str))
    rows.drop(columns='split').to_csv(output / 'metadata.csv', index=False)
    rows.to_csv(output / 'split_manifest.csv', index=False)
    (output / 'class_mapping.json').write_text(json.dumps({
        'class_to_id': mapping, 'id_to_class': {str(i): n for n, i in mapping.items()}}, indent=2))
    summary = {'source': str(source.resolve()), 'excluded_R': int((~keep).sum()),
               'classes': names, 'recordings': len(rows),
               'split_counts': rows.groupby(['participant', 'split']).size().unstack(fill_value=0).to_dict('index'),
               'original_training': int(rows.split.eq('train').sum()),
               'training_with_mirrors': int(2 * rows.split.eq('train').sum()),
               'seed_validation': 42, 'seed_test': 43,
               'test_note': 'Held-out recordings of familiar signers. Some were training clips of the old model; do not compare old-model test accuracy.',
               'mirroring': 'Trainer adds one whole-sequence reflection per training recording after splitting; evaluation originals stay untouched.'}
    (output / 'preparation_report.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
