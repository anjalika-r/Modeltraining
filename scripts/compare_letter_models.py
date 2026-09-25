"""Compare two saved models on the expanded run's identical validation clips."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sign_features import model_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    candidate_config = json.loads((args.candidate / 'run_config.json').read_text())
    data = Path(candidate_config['data_dir'])
    manifest = pd.read_csv(args.candidate / 'split_manifest.csv')
    validation = manifest.split.eq('validation').to_numpy()
    x = np.load(data / candidate_config['input_file'])[validation]
    rows = manifest.loc[validation].copy().reset_index(drop=True)
    labels = rows.class_label.to_numpy()
    results = {}
    predictions = {}
    for name, directory in [('three_signers', args.reference), ('four_signers', args.candidate)]:
        config = json.loads((directory / 'run_config.json').read_text())
        contract = json.loads((directory / 'inference_config.json').read_text())
        training_manifest = pd.read_csv(directory / 'split_manifest.csv')
        original_x = np.load(Path(config['data_dir']) / config['input_file'])
        train_hashes = {hashlib.sha256(a.tobytes()).hexdigest() for a in original_x[training_manifest.split.eq('train')]}
        if any(hashlib.sha256(a.tobytes()).hexdigest() in train_hashes for a in x):
            raise ValueError(f'Evaluation tensor overlaps {name} training data')
        if name == 'three_signers':
            old_val = training_manifest.split.eq('validation')
            keys = ['participant', 'source_file']
            aligned = rows.reset_index().merge(training_manifest.loc[old_val, keys].assign(old_index=np.flatnonzero(old_val)), on=keys)
            if len(aligned) != old_val.sum() or not np.array_equal(x[aligned['index']], original_x[aligned.old_index]):
                raise ValueError('Historical validation clips are missing or changed')
        model = tf.keras.models.load_model(directory / 'best_lstm_model.keras')
        probabilities = model.predict(model_features(x, contract['schema']), verbose=0)
        predictions[name] = np.asarray(contract['class_names'])[probabilities.argmax(axis=1)]
        rows[name + '_prediction'] = predictions[name]
        rows[name + '_correct'] = predictions[name] == labels
        groups = {'all_validation': np.ones(len(rows), dtype=bool),
                  'original_three_signers': rows.participant.ne('alix30').to_numpy(),
                  'alix30': rows.participant.eq('alix30').to_numpy()}
        results[name] = {group: dict(clips=int(mask.sum()),
                                    accuracy=float(accuracy_score(labels[mask], predictions[name][mask])),
                                    macro_f1=float(f1_score(labels[mask], predictions[name][mask], average='macro', zero_division=0)))
                         for group, mask in groups.items()}
    rows.to_csv(args.candidate / 'comparison_predictions.csv', index=False)
    for field in ['participant', 'letter']:
        grouped = rows.groupby(field).agg(clips=('class_label', 'size'),
                    three_signers_accuracy=('three_signers_correct', 'mean'),
                    four_signers_accuracy=('four_signers_correct', 'mean'))
        grouped['change_percentage_points'] = 100 * (grouped.four_signers_accuracy - grouped.three_signers_accuracy)
        grouped.to_csv(args.candidate / f'comparison_by_{field}.csv')
    (args.candidate / 'comparison_summary.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
