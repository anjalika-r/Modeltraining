"""Run a fixed-split, three-seed comparison of class versus signer-letter weighting."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs' / 'signer_balance_experiment'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    reference = pd.read_csv(root / 'models/letters_four_signers/split_manifest.csv')
    # Reuse the already completed seed-42 class-balanced control.
    for weighting in ['class', 'signer-letter']:
        for seed in [42, 43, 44]:
            directory = root / 'models' / (f'letters_four_{weighting}_seed{seed}'
                                          if (weighting, seed) != ('class', 42) else 'letters_four_signers')
            if not (directory / 'evaluation_summary.json').exists():
                command = [sys.executable, '-u', str(root / 'scripts/train_lstm_sign_model_tunable.py'),
                           '--data-dir', str(root / 'data/letters_four_signers'),
                           '--output-dir', str(directory), '--feature-set', 'baseline',
                           '--participants', 'rithika', 'angel', 'alix30', 'mathur',
                           '--previous-split-manifest', str(root / 'models/letters_four_signers/split_manifest.csv'),
                           '--seed', str(seed), '--weighting', weighting, '--epochs', '100']
                print(f'Training {weighting} seed {seed}', flush=True)
                env = dict(os.environ, MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
                with (output / f'{weighting}_{seed}.log').open('w', encoding='utf-8') as log:
                    subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            manifest = pd.read_csv(directory / 'split_manifest.csv')
            pd.testing.assert_frame_equal(reference[['participant', 'source_file', 'class_id', 'split']],
                                          manifest[['participant', 'source_file', 'class_id', 'split']])
            predictions = pd.read_csv(directory / 'validation_predictions.csv')
            accuracy = predictions.groupby('participant').correct.mean()
            row = dict(weighting=weighting, seed=seed, model_dir=str(directory),
                       overall_accuracy=float(predictions.correct.mean()),
                       mean_signer_accuracy=float(accuracy.mean()),
                       worst_signer_accuracy=float(accuracy.min()), **accuracy.to_dict())
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / 'runs.csv', index=False)
            print(json.dumps(row), flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby('weighting')[['overall_accuracy', 'mean_signer_accuracy', 'worst_signer_accuracy',
                                        'rithika', 'angel', 'mathur', 'alix30']].agg(['mean', 'std'])
    summary.to_csv(output / 'method_summary.csv')
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
