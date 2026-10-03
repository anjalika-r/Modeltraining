# AIRV mirrored model without R

Trained on Angel, Rithika, Mathur and Isaac, excluding all 96 R recordings. Original source dataset is preserved. The model supports 23 letters; H, J and R are absent.

1,168 original training recordings plus 1,168 whole-sequence mirrored copies; 244 validation and 206 held-out test recordings. Split assignments are frozen before augmentation. Checkpoint selected at epoch 53; training stopped after 68 epochs.

## Paired validation comparison

| Signer | Clips | Current mirrored model | AIRV mirrored model |
|---|---:|---:|---:|
| angel | 84 | 80.95% | 88.10% |
| rithika | 73 | 78.08% | 73.97% |
| mathur | 49 | 77.55% | 81.63% |
| isaac | 38 | 55.26% | 89.47% |

## Candidate evaluation

| Evaluation | Clips | Accuracy | Accepted at 85% | Accuracy among accepted |
|---|---:|---:|---:|---:|
| validation original | 244 | 82.79% | 142 | 96.48% |
| validation mirrored | 244 | 80.33% | 145 | 95.17% |
| test original | 206 | 75.73% | 109 | 93.58% |
| test mirrored | 206 | 76.70% | 118 | 90.68% |

## Interpretation

Validation compares identical clips without filtering or renormalizing the old model's 24-class output. The experiment changes training data, removes R and reserves test recordings, so gains cannot be attributed solely to adding Isaac. Rithika validation regresses. At 85%, the new model accepts fewer validation attempts (142 versus 160), but accepted accuracy improves (96.48% versus 90.00%).

Synthetic mirroring tests reflections of landmark sequences; actual signing with each hand still needs a camera check. Test recordings were unused for fitting and checkpoint selection in this run, but familiar signers are represented in training. Some test recordings were used to train the older model; no paired test comparison is reported. Isaac has no C recordings. Removing R resolves the R/S conflicting-label group; two identical S tensors remain together within one split.

## Reproduce

```powershell
python scripts/prepare_airv_experiment.py
python scripts/train_lstm_sign_model_tunable.py --data-dir data/AIRV_no_r --output-dir models/letters_airv_no_r_mirrored --feature-set baseline --split-manifest data/AIRV_no_r/split_manifest.csv --mirror-augmentation --epochs 100
python scripts/evaluate_airv_experiment.py
python scripts/predict_webcam.py --model-dir models/letters_airv_no_r_mirrored --threshold 0.85
```

Preparation and training refuse existing output directories; use fresh output paths when rerunning. Evaluation uses the fixed experiment paths. The website remains on its existing checkpoint.
