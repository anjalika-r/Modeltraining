"""Summarize the fixed-split experiment and select a validation candidate."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs/signer_balance_experiment'
    runs = pd.read_csv(output / 'runs.csv')
    if set(zip(runs.weighting, runs.seed)) != {(w, s) for w in ['class', 'signer-letter'] for s in [42, 43, 44]}:
        raise ValueError('Complete both methods and all three seeds before selecting')
    # Compare methods over all seeds, then favor the weakest signer when choosing a checkpoint.
    method = runs.groupby('weighting').mean_signer_accuracy.mean().idxmax()
    selected = runs[runs.weighting == method].sort_values(
        ['worst_signer_accuracy', 'mean_signer_accuracy'], ascending=False).iloc[0]
    selection = dict(method=method, seed=int(selected.seed), model_dir=selected.model_dir,
                     method_criterion='highest mean equal-signer validation accuracy across seeds 42/43/44',
                     checkpoint_criterion='highest minimum signer accuracy within selected method; tie-break mean signer accuracy',
                     limitation='Validation used for training and selection; not an independent test estimate.')
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    signers = ['alix30', 'angel', 'mathur', 'rithika']
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, weighting, label in [(-0.13, 'class', 'Letter balancing'), (0.13, 'signer-letter', 'Signer + letter balancing')]:
        values = runs[runs.weighting == weighting][signers] * 100
        positions = np.arange(4) + offset
        ax.errorbar(positions, values.mean(), yerr=values.std(), fmt='o', capsize=5, label=label)
        for _, row in values.iterrows():
            ax.scatter(positions, row, s=18, alpha=0.4)
    ax.set_xticks(np.arange(4), signers)
    ax.set_ylabel('Validation accuracy (%)')
    ax.set_title('Same validation clips, three training seeds\nDots: runs; bars: mean ± sample standard deviation')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'signer_accuracy.png', dpi=160)
    plt.close(fig)
    old = pd.read_csv(root / 'models/letters_four_signers/comparison_by_participant.csv').set_index('participant')
    lines = ['# Signer balancing experiment', '',
             'All runs use the same 1,691 training clips and 299 validation clips, baseline features, architecture, and training settings. Original Alix and Aldebaran clips remain excluded. Only training seed and weighting vary. Seed-42 class balancing reuses the earlier completed model.', '',
             'Signer-letter weights give every signer 25% of training weight and each of their 24 letters equal weight. Existing class weights already gave Rithika 24.99%; this is not simply an increase in Rithika weighting. Checkpoints within runs still minimize pooled validation loss to isolate the weighting change.', '',
             '| Method | Seed | Overall | Mean signer | Worst signer | Alix30 | Angel | Mathur | Rithika |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in runs.itertuples():
        metrics = [row.overall_accuracy, row.mean_signer_accuracy, row.worst_signer_accuracy,
                   row.alix30, row.angel, row.mathur, row.rithika]
        lines.append(f'| {row.weighting} | {row.seed} | ' + ' | '.join(f'{v:.2%}' for v in metrics) + ' |')
    lines += ['', 'Method means ± sample standard deviation:', '', '| Method | Mean signer accuracy | Rithika accuracy |', '| --- | ---: | ---: |']
    for name, group in runs.groupby('weighting'):
        lines.append(f'| {name} | {group.mean_signer_accuracy.mean():.2%} ± {group.mean_signer_accuracy.std():.2%} | {group.rithika.mean():.2%} ± {group.rithika.std():.2%} |')
    lines += ['', f"Selected candidate: `{selected.model_dir}`.", '',
              'Selection uses the highest mean equal-signer validation accuracy across seeds to choose a method, then the highest minimum signer accuracy to choose its candidate. Ties use mean signer accuracy. This favors performance across all four signers.', '',
              '| Signer | Original three-signer model | Initial four-signer model | Selected candidate |', '| --- | ---: | ---: | ---: |']
    for signer in signers:
        lines.append(f'| {signer} | {old.loc[signer, "three_signers_accuracy"]:.2%} | {old.loc[signer, "four_signers_accuracy"]:.2%} | {selected[signer]:.2%} |')
    lines += ['', 'These are exploratory validation comparisons, not independent test scores. Three seeds measure some training variation but do not establish statistical significance or explain the original regression conclusively. The 77 Rithika validation clips are unchanged. The inherited mixed resampling and conflicting training labels remain unchanged to isolate weighting. Validate the selected candidate on fresh recordings and in the webcam before claiming generalization.', '',
              '```powershell', 'python scripts/run_signer_balance_experiment.py',
              'python scripts/summarize_signer_balance_experiment.py',
              f'python scripts/predict_webcam.py --model-dir "{selected.model_dir}"', '```', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
