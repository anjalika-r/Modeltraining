"""Compare baseline and shape-contact AIRV models on frozen recording splits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sign_features import mirror_body_hands, model_features


def metrics(frame):
    accepted = frame.confidence.ge(.85)
    return {'clips': len(frame), 'accuracy': float(frame.correct.mean()),
            'accepted': int(accepted.sum()),
            'accepted_accuracy': float(frame.loc[accepted, 'correct'].mean()) if accepted.any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('models/letters_airv_no_r_mirrored'))
    parser.add_argument('--candidate', type=Path, default=Path('models/letters_airv_no_r_shape_contact'))
    args = parser.parse_args()
    manifests, configurations, datasets = {}, {}, {}
    for name, directory in [('baseline', args.baseline), ('shape_contact', args.candidate)]:
        configurations[name] = json.loads((directory / 'run_config.json').read_text())
        manifests[name] = pd.read_csv(directory / 'split_manifest.csv')
        datasets[name] = np.load(Path(configurations[name]['data_dir']) / configurations[name]['input_file'])
    keys = ['participant', 'source_file', 'class_label', 'split']
    pd.testing.assert_frame_equal(manifests['baseline'][keys], manifests['shape_contact'][keys])
    np.testing.assert_array_equal(datasets['baseline'], datasets['shape_contact'])
    unchanged = ['mirror_augmentation', 'seed', 'weighting', 'batch_size', 'learning_rate',
                 'epochs', 'patience', 'lr_patience', 'lr_factor', 'min_lr',
                 'frame_dense_units', 'lstm_units', 'dense_units', 'dropout',
                 'recurrent_dropout', 'bidirectional', 'l2']
    if any(configurations['baseline'][key] != configurations['shape_contact'][key] for key in unchanged):
        raise ValueError('Training settings differ beyond the feature set')
    rows, x = manifests['baseline'], datasets['baseline']
    train = x[rows.split.eq('train')]
    training_hashes = {hashlib.sha256(a.tobytes()).hexdigest()
                       for a in np.concatenate([train, mirror_body_hands(train)])}
    results, predictions = {}, []
    for name, directory in [('baseline', args.baseline), ('shape_contact', args.candidate)]:
        contract = json.loads((directory / 'inference_config.json').read_text())
        model = tf.keras.models.load_model(directory / 'best_lstm_model.keras', compile=False)
        results[name] = {}
        for split in ['validation', 'test']:
            subset = rows.loc[rows.split.eq(split)].copy().reset_index(drop=True)
            original = x[rows.split.eq(split)]
            for orientation, sequences in [('original', original), ('mirrored', mirror_body_hands(original))]:
                if any(hashlib.sha256(a.tobytes()).hexdigest() in training_hashes for a in sequences):
                    raise ValueError('Original or mirrored evaluation overlaps training')
                probs = model(model_features(sequences, contract['schema']), training=False).numpy()
                frame = subset.assign(model=name, orientation=orientation,
                                      predicted_letter=np.array(contract['class_names'])[probs.argmax(axis=1)],
                                      confidence=probs.max(axis=1))
                frame['correct'] = frame.predicted_letter.eq(frame.class_label)
                results[name][split + '_' + orientation] = {
                    'all': metrics(frame),
                    'by_letter': {letter: metrics(group) for letter, group in frame.groupby('class_label')},
                    'by_participant': {signer: metrics(group) for signer, group in frame.groupby('participant')}}
                predictions.append(frame)
    summary = {'checks': {'identical_recordings_and_splits': True,
                          'same_hyperparameters_and_mirroring': True,
                          'no_original_or_mirrored_training_overlap': True},
               'results': results,
               'limitations': ['Test recordings were inspected during earlier error analysis; this follow-up test comparison is exploratory, not a fresh independent final test.',
                               'All signers appear in training; results do not establish unseen-user accuracy.',
                               'Mirrored landmarks are synthetic; camera performance with each hand needs checking.']}
    (args.candidate / 'feature_comparison.json').write_text(json.dumps(summary, indent=2))
    pd.concat(predictions, ignore_index=True).to_csv(args.candidate / 'feature_comparison_predictions.csv', index=False)
    print(json.dumps({name: {key: value['all'] for key, value in groups.items()}
                      for name, groups in results.items()}, indent=2))


if __name__ == '__main__':
    main()
