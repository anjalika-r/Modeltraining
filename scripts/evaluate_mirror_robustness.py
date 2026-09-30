"""Compare models on identical original and synthetic mirrored validation clips."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sign_features import mirror_body_hands, model_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    results = {}
    expected_x = expected_y = None
    for name, directory in [('reference', args.reference), ('candidate', args.candidate)]:
        config = json.loads((directory / 'run_config.json').read_text())
        contract = json.loads((directory / 'inference_config.json').read_text())
        manifest = pd.read_csv(directory / 'split_manifest.csv')
        mask = manifest.split.eq('validation').to_numpy()
        x = np.load(Path(config['data_dir']) / config['input_file'])[mask]
        y = manifest.loc[mask, 'class_label'].to_numpy()
        if expected_x is not None:
            if not np.array_equal(x, expected_x) or not np.array_equal(y, expected_y):
                raise ValueError('Models must use identical validation recordings in identical order')
        expected_x, expected_y = x, y
        model = tf.keras.models.load_model(directory / 'best_lstm_model.keras', compile=False)
        results[name] = {}
        for orientation, frames in [('original', x), ('synthetic_mirrored', mirror_body_hands(x))]:
            probs = model.predict(model_features(frames, contract['schema']), verbose=0)
            pred = np.asarray(contract['class_names'])[probs.argmax(axis=1)]
            results[name][orientation] = {
                'clips': len(y), 'accuracy': float(np.mean(pred == y)),
                'per_letter': {
                    letter: {'clips': int(np.sum(y == letter)),
                             'accuracy': float(np.mean(pred[y == letter] == letter))}
                    for letter in sorted(set(y))},
            }
    output = args.candidate / 'mirror_robustness.json'
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps({name: {orientation: result['accuracy']
                            for orientation, result in orientations.items()}
                      for name, orientations in results.items()}, indent=2))
    print('Synthetic reflections are diagnostic; real opposite-handed captures still need evaluation.')


if __name__ == '__main__':
    main()
