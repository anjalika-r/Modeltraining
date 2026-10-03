# AIRV shape-contact experiment

The shape-contact model uses the exact baseline recording tensors and train/validation/test assignments, seed 42, class weighting, architecture widths, optimizer and early-stopping settings. Only the feature representation changes from 227 to 497 inputs per frame. This increases the projection layer parameter count. Training remains 1,168 originals plus 1,168 whole-sequence mirrors. R stays excluded; there are 23 supported letters.

Selected epoch 53 of 68. Validation loss: 0.35779.

## Overall comparison

| Evaluation | Clips | Baseline accuracy | Shape-contact accuracy |
|---|---:|---:|---:|
| validation original | 244 | 82.79% | 86.48% |
| validation mirrored | 244 | 80.33% | 88.11% |
| test original | 206 | 75.73% | 84.95% |
| test mirrored | 206 | 76.70% | 85.92% |

## Decisions at 85% score

| Evaluation | Baseline accepted | Baseline accepted accuracy | Shape-contact accepted | Shape-contact accepted accuracy |
|---|---:|---:|---:|---:|
| validation original | 142/244 | 96.48% | 182/244 | 97.25% |
| test original | 109/206 | 93.58% | 149/206 | 97.32% |

## Target letters

Correct predictions on original recordings. Per-letter sample sizes are small.

| Letter | Validation baseline | Validation shape-contact | Test baseline | Test shape-contact |
|---|---:|---:|---:|---:|
| E | 12/13 | 13/13 | 7/12 | 10/12 |
| I | 5/9 | 6/9 | 3/8 | 5/8 |
| K | 10/13 | 13/13 | 8/10 | 9/10 |
| Q | 7/7 | 7/7 | 4/5 | 5/5 |
| Y | 9/9 | 9/9 | 5/7 | 7/7 |
| U | 18/18 | 18/18 | 14/16 | 14/16 |

I remains a weakness: the shape-contact model accepts six validation I attempts at 85%, and two are incorrect. U keeps 18/18 validation accuracy while acceptance rises from 8/18 to 17/18.

## Validation by signer

| Signer | Baseline accuracy | Shape-contact accuracy |
|---|---:|---:|
| angel | 88.10% | 84.52% |
| rithika | 73.97% | 84.93% |
| mathur | 81.63% | 85.71% |
| isaac | 89.47% | 94.74% |

Angel validation accuracy decreases from 88.10% to 84.52%; the other three improve. Recommend testing this candidate live before changing the website.

## Limits

Test clips were inspected during earlier error analysis and influenced the choice of this experiment. Their follow-up results are exploratory and are not a fresh independent final evaluation. All signers appear in training. Synthetic mirrors do not establish live opposite-handed performance. The 85% threshold is an uncalibrated model score. Feature and augmentation tests pass, and the comparison verifies identical tensors, splits and training settings plus no evaluation overlap with original or mirrored training tensors.

## Run

```powershell
python scripts/train_lstm_sign_model_tunable.py --data-dir data/AIRV_no_r --output-dir models/letters_airv_no_r_shape_contact --feature-set shape-contact --split-manifest data/AIRV_no_r/split_manifest.csv --mirror-augmentation --epochs 100
python scripts/compare_airv_features.py
python scripts/predict_webcam.py --model-dir models/letters_airv_no_r_shape_contact --threshold 0.85
```

The completed training directory is preserved; choose a fresh output directory to train another run. The website was not changed.
