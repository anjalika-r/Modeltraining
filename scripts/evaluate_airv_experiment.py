"""Evaluate AIRV checkpoint once on frozen test and validation recordings."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sign_features import mirror_body_hands, model_features


def scores(labels, predictions, confidence):
    accepted = confidence >= .85
    return {'clips': len(labels), 'accuracy': float(np.mean(labels == predictions)),
            'accepted_at_85_percent': int(accepted.sum()),
            'accepted_accuracy': float(np.mean(labels[accepted] == predictions[accepted])) if accepted.any() else None}


def main():
    candidate = Path('models/letters_airv_no_r_mirrored')
    source = Path('data/AIRV_no_r')
    rows = pd.read_csv(candidate / 'split_manifest.csv')
    x = np.load(source / 'sequences_body_hands.npy')
    train = x[rows.split.eq('train')]
    hashes = {hashlib.sha256(a.tobytes()).hexdigest()
              for a in np.concatenate([train, mirror_body_hands(train)])}
    results, tables = {}, []
    for name, directory in [('current_model', Path('models/letters_three_signers_mirrored')),
                            ('airv_model', candidate)]:
        config = json.loads((directory / 'inference_config.json').read_text())
        model = tf.keras.models.load_model(directory / 'best_lstm_model.keras', compile=False)
        results[name] = {}
        for split in (['validation'] if name == 'current_model' else ['validation', 'test']):
            mask = rows.split.eq(split).to_numpy()
            subset = rows.loc[mask].copy().reset_index(drop=True)
            sequences = x[mask]
            for a in sequences:
                if hashlib.sha256(a.tobytes()).hexdigest() in hashes:
                    raise ValueError('Candidate training overlaps evaluation')
            if name == 'current_model':
                old_config = json.loads((directory / 'run_config.json').read_text())
                old_rows = pd.read_csv(directory / 'split_manifest.csv')
                old_x = np.load(Path(old_config['data_dir']) / old_config['input_file'])
                old_train = old_x[old_rows.split.eq('train')]
                old_hashes = {hashlib.sha256(a.tobytes()).hexdigest()
                              for a in np.concatenate([old_train, mirror_body_hands(old_train)])}
                if any(hashlib.sha256(a.tobytes()).hexdigest() in old_hashes for a in sequences):
                    raise ValueError('Current model training overlaps validation comparison')
            for orientation, values in [('original', sequences), ('mirrored', mirror_body_hands(sequences))]:
                probabilities = model(model_features(values, config['schema']), training=False).numpy()
                predictions = np.array(config['class_names'])[probabilities.argmax(axis=1)]
                confidence = probabilities.max(axis=1)
                labels = subset.class_label.to_numpy()
                key = split + '_' + orientation
                results[name][key] = {'all': scores(labels, predictions, confidence),
                                      'by_participant': {}, 'by_letter': {}}
                familiar = subset.participant.ne('isaac').to_numpy()
                results[name][key]['original_three_signers'] = scores(
                    labels[familiar], predictions[familiar], confidence[familiar])
                for field, destination in [('participant', 'by_participant'), ('class_label', 'by_letter')]:
                    for group in subset[field].unique():
                        select = subset[field].eq(group).to_numpy()
                        results[name][key][destination][group] = scores(labels[select], predictions[select], confidence[select])
                tables.append(subset.assign(model=name, orientation=orientation,
                                            predicted_letter=predictions, confidence=confidence,
                                            accepted=confidence >= .85, correct=predictions == labels))
    pd.concat(tables, ignore_index=True).to_csv(candidate / 'paired_evaluation_predictions.csv', index=False)
    results['limitations'] = ['Mirrored evaluation is synthetic and does not establish real left-handed camera performance.',
                             'Test recordings were held out from this run, but some were training data for the current model; current-model test comparisons are omitted.',
                             'All test signers also occur in training; this does not measure unseen-signer accuracy.',
                             'Isaac has no C recordings. Added Isaac data cannot directly teach his C sign.']
    (candidate / 'mirror_and_threshold_evaluation.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({name: {key: value['all'] for key, value in parts.items()}
                      for name, parts in results.items() if isinstance(parts, dict)}, indent=2))


if __name__ == '__main__':
    main()
