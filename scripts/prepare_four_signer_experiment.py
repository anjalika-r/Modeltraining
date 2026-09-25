"""Prepare merged data while excluding original Alix/Aldebaran identities."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    meta = pd.read_csv(args.data_dir / 'metadata.csv')
    original = meta['participant_original'].fillna(meta['participant'])
    keep = meta.participant.isin(['angel', 'rithika', 'mathur', 'alix30']) & ~original.isin(['alix', 'aldebaran'])
    selected = meta.loc[keep].copy()
    # New exports name their actual resampled timestamps output_timestamps.
    selected['selected_timestamps'] = selected.selected_timestamps.fillna(selected.output_timestamps)
    x = np.load(args.data_dir / 'sequences_body_hands.npy')[keep]
    y = np.load(args.data_dir / 'labels.npy')[keep]
    hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in x]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected['sequence_sha256'] = hashes
    selected.to_csv(args.output_dir / 'metadata.csv', index=False)
    np.save(args.output_dir / 'sequences_body_hands.npy', x)
    np.save(args.output_dir / 'labels.npy', y)
    np.save(args.output_dir / 'labels_text.npy', selected.class_label.to_numpy(dtype=str))
    shutil.copy2(args.data_dir / 'class_mapping.json', args.output_dir / 'class_mapping.json')
    report = dict(source=str(args.data_dir.resolve()), sequences=len(selected), duplicate_tensor_rows=len(hashes)-len(set(hashes)),
                  participants=selected.participant.value_counts().to_dict(),
                  excluded_original_identities=original.loc[~keep].value_counts().to_dict(),
                  resampling_methods=selected.resampling_method.fillna('legacy_uniform_real_frames').value_counts().to_dict())
    (args.output_dir / 'preparation_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
