# Mirrored sign model

This checkpoint recognises all 24 supported letters using training examples in
both original and mirrored orientations. H and J are not in the dataset.

`best_lstm_model.keras` is the selected checkpoint; `inference_config.json`
defines its preprocessing and labels. Use the repository's shared predictor:

```powershell
python scripts/predict_webcam.py --model-dir models/letters_three_signers_mirrored
```

Install the repository dependencies first. The predictor also requires
`models/holistic_landmarker.task` (see the main README for setup).

Training used 1,245 original sequences plus 1,245 mirrored copies, with the
existing 220-recording validation split. The selected checkpoint is epoch 87.

| Validation input | Previous model | This model |
| --- | ---: | ---: |
| Original recordings | 80.9% | 78.2% |
| Synthetic mirrored recordings | 43.2% | 78.6% |

Per-letter results are in `mirror_robustness.json`. Synthetic reflections do
not replace evaluation on real opposite-handed recordings. The website must
be configured to load this model directory to use this checkpoint.
