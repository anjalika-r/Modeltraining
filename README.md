# Signsense: Auslan letter recognition

Train an LSTM on normalized MediaPipe body/hand sequences, then show predicted
letters in a webcam preview. Run commands from the repository root.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/download_landmarker.py
```

Use the virtual environment's Python for the commands below, or activate it.
The implementation was exercised on Python 3.13, TensorFlow 2.21.0 and MediaPipe
0.10.33 on Windows CPU. It uses the current MediaPipe Tasks Holistic API, not
the removed `mp.solutions` API. The downloader obtains Google's versioned
[Holistic model](https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/holistic-landmarker.ts).
Camera frames are processed locally.

## Train

New training runs default to `--feature-set shape-contact`: 497 inputs per
frame comprising the original body/hand context, 63 palm-normalized coordinates,
15 joint angles divided by pi and 5 extension ratios per hand, 100 cross-hand
XY distances, wrist XY displacement, and two geometry-valid flags. Cross-hand
points are `[0, 4, 8, 12, 16, 20, 5, 9, 13, 17]` in left-major order, covering
the wrist, fingertips and finger bases. Distances use shoulder-normalized XY;
independent hand depths are not treated as shared 3D coordinates.

The input dataset stays at 227 features; training and webcam inference share
the same transformation. Missing or degenerate hands have their coordinates
and derived blocks zeroed; cross-hand values require both hands to be valid.
Original detection flags and separate geometry-valid flags distinguish missing
information. These checks cannot identify plausible but incorrect tracking,
so they do not fix the reported L/M/N/R tracking problems. No interpolation,
motion deltas, soft-contact threshold, or handedness swapping is applied.

Use `--feature-set baseline` to reproduce the original representation. Existing
227-input checkpoints remain supported. Compare new runs on the same recording
split; the historical scores below refer to the original baseline features,
not the new representation. Improvement requires training and evaluation.

Example feature experiment matching the existing three-signer split:

```powershell
python scripts/train_lstm_sign_model_tunable.py --data-dir data/letter_dataset --output-dir models/letters_shape_contact --feature-set shape-contact --participants angel rithika mathur --validation-fraction 0.15 --epochs 100
python scripts/predict_webcam.py --model-dir models/letters_shape_contact
```

For the three-signer webcam experiment, train on 85% of Angel, Rithika and
Mathur's recordings and validate on the remaining 15%:

```powershell
python scripts/train_lstm_sign_model_tunable.py --data-dir data/letter_dataset --output-dir models/letters_three_signers --participants angel rithika mathur --validation-fraction 0.15 --epochs 100
```

This mode stratifies source recordings by participant and letter with seed 42.
All frames and repeated rows of a source recording stay together. Alix and
Aldebaran are marked `excluded` in the split manifest and are not used for
training, validation, or testing. There is no independent test score in this
mode: validation measures new recordings of familiar signers. Source paths are
the grouping identifiers; derived recordings stored under different paths need
a shared source identifier before using this split.

Try the resulting model with:

```powershell
python scripts/predict_webcam.py --model-dir models/letters_three_signers
```

For reruns, choose a new output directory. The earlier participant-held-out
baseline remains in `models/letters_baseline`.

The completed three-signer run used 1,245 training clips and 220 validation
clips, excluding 133 clips. It stopped after 79 epochs and selected epoch 64,
with **80.91% validation accuracy** and 0.6408 validation loss. This is a
familiar-signer validation result, not an unseen-user accuracy estimate, and
is not directly comparable to the earlier participant-held-out score.

The supplied `data/letter_dataset` is already prepared. To regenerate it into
a **new** directory:

```powershell
python scripts/prepare_letter_dataset.py --data-dir data/processed_18_frames --output-dir data/letters_new
```

Train with separate validation and test participants:

```powershell
python scripts/train_lstm_sign_model_tunable.py --data-dir data/letter_dataset --output-dir models/letters_run2 --val-participant rithika --test-participant mathur --epochs 100
```

Choose a new output directory for each run. Test samples are excluded from
training and early stopping. Omitting `--test-participant` trains on everyone
except the validation participant, but then there is no independent test score.
`train_lstm_sign_model.py` is a compatibility entry point to the same trainer.

Each run saves the selected checkpoint, class mapping, inference configuration,
split manifest, training curves, classification reports and validation
predictions. If a test participant is supplied, it also saves a test report.
Keep the entire model directory together for inference.

## Try the webcam

### Four-signer merged-data experiment (2026-09-25)

The separate model in `models/letters_four_signers` uses Angel, Rithika,
Mathur and the 525 new Alix30 clips. Preparation excludes 114 Aldebaran clips
and one original Alix clip that the merged metadata had renamed to Alix30.
The supplied merged dataset and earlier models are preserved.

This run matches `letters_three_signers` baseline features (227 inputs),
architecture, seed 42 and optimizer settings. It retains all 1,245 original
training and 220 validation recordings, adding 446 training and 79 validation
Alix30 clips. Early stopping selected epoch 74 of 89, with validation loss
0.66888. Both checkpoints were evaluated on the same 299 validation clips:

| Validation group | Clips | Three-signer model | Four-signer model |
| --- | ---: | ---: | ---: |
| Original three signers | 220 | 80.91% | 77.73% |
| Alix30 | 79 | 2.53% | 78.48% |
| All four signers | 299 | 60.20% | 77.93% |

Per-signer accuracy changed from 83.15% to 84.27% for Angel, 77.78% to
79.63% for Mathur, and 80.52% to 68.83% for Rithika. The added data greatly
improves Alix30 recognition but reduces Rithika accuracy. These are
validation results used for checkpoint selection, not independent test
results or evidence of unseen-signer generalization. Alix30 was unseen by
the old model but represented in the new model's training set.

The comparison verifies that historical validation tensors are unchanged and
that no exact evaluation tensor appears in either model's training data.
Four identical older tensors with conflicting R/S labels remain entirely in
training to preserve the historical experiment. New Alix30 data includes 392
linearly interpolated sequences and 133 sequences sampled from real frames;
preparation uses their `output_timestamps` when `selected_timestamps` is absent.
Live inference still uses uniform real-frame sampling; webcam performance
needs checking. H and J remain absent.

Reproduce using fresh output directories:

```powershell
python scripts/prepare_four_signer_experiment.py --data-dir data/merged_letter_dataset_unique/merged_letter_dataset --output-dir data/letters_four_signers
python scripts/train_lstm_sign_model_tunable.py --data-dir data/letters_four_signers --output-dir models/letters_four_signers --feature-set baseline --participants rithika angel alix30 mathur --previous-split-manifest models/letters_three_signers/split_manifest.csv --validation-fraction 0.15 --epochs 100
python scripts/compare_letter_models.py --reference models/letters_three_signers --candidate models/letters_four_signers
python scripts/predict_webcam.py --model-dir models/letters_four_signers
```

The model directory includes `comparison_summary.json`,
`comparison_by_participant.csv`, `comparison_by_letter.csv`, and paired
`comparison_predictions.csv`, alongside the normal training artifacts.

### Follow-up: signer balancing and training variation

The follow-up compared ordinary letter balancing with `--weighting signer-letter`
using seeds 42, 43 and 44. Every run preserves the entire initial four-signer
split via `--previous-split-manifest models/letters_four_signers/split_manifest.csv`;
changing the training seed does not change validation recordings. Architecture,
baseline features and checkpoint selection by pooled validation loss stay fixed.
Signer-letter weights give each signer equal total weight and each of their
letters equal weight within that signer. Weights use training counts only.

Across three seeds, mean equal-signer validation accuracy was **76.25%** for
ordinary letter balancing versus **74.55%** for signer-letter balancing.
Mean Rithika accuracy was **70.56% for both methods**; its range was
64.94–77.92% for ordinary balancing and 68.83–71.43% for signer-letter balancing.
Existing class balancing already gave Rithika 24.99% of training weight.
These runs do not support signer imbalance alone as the explanation or show
that signer balancing fixes the regression.

The recommended candidate is **`models/letters_four_class_seed43`**. Selection
first favors the method with highest mean equal-signer accuracy across seeds,
then its run with the highest minimum signer accuracy. It selected epoch 84
of 99, with validation loss 0.62462:

| Validation group | Initial four-signer model | Recommended candidate |
| --- | ---: | ---: |
| Rithika (77 clips) | 68.83% | 77.92% |
| Angel (89 clips) | 84.27% | 83.15% |
| Mathur (54 clips) | 79.63% | 83.33% |
| Alix30 (79 clips) | 78.48% | 79.75% |
| Combined (299 clips) | 77.93% | 80.94% |

On the original 220 validation clips, this candidate scores 81.36%, compared
with the historical three-signer model's 80.91%. Rithika individually remains
below the historical 80.52% (60 correct versus 62 of 77). These validation
results were used for selection; fresh recordings are needed for independent
evaluation. No existing model was replaced, and original Alix/Aldebaran remain
excluded. The conflicting legacy training labels and mixed resampling remain
unchanged to keep this comparison focused on weighting and seed variation.

```powershell
python scripts/predict_webcam.py --model-dir models/letters_four_class_seed43
```

Full run results, a per-signer plot and selection details are saved under
`outputs/signer_balance_experiment/` in `report.md`, `runs.csv`,
`method_summary.csv`, `signer_accuracy.png` and `selection.json`.
Reproduce or summarize the experiment with:

```powershell
python scripts/run_signer_balance_experiment.py
python scripts/summarize_signer_balance_experiment.py
```

The runner reuses completed runs, including the initial seed-42 control;
use fresh output paths for a new experiment. Per-run weight totals are saved
in `training_weight_summary.csv` for newly trained models.

A baseline has been trained locally in `models/letters_baseline` (ignored by Git).

```powershell
python scripts/predict_webcam.py --model-dir models/letters_baseline
```

Keep shoulders and signing hands visible. Hold/perform one letter over roughly
three seconds. The demo samples 18 frames across that window and predicts every
0.25 seconds. It displays a letter after three consecutive predictions agree
and exceed the default 0.8 softmax threshold. Otherwise it shows “Uncertain”,
the best guess and its score, and whether confidence or agreement is missing.
The top three guesses are shown in the preview and terminal for diagnosis.
Missing shoulders/hands reset the window so old signs are not carried through
tracking loss. `R` resets manually; `Q` or Escape exits. Try `--camera 1` if the
default camera is unavailable.

The preview is mirrored for convenience; model input is unmirrored by default.
The original recording app's mirroring convention is not present in this repo.
If it mirrored images **before landmark extraction**, use `--mirror-input` to
match it. MediaPipe model/version differences from the recording app may also
affect predictions even though normalization matches exactly.

Recorded video also works:

```powershell
python scripts/predict_webcam.py --model-dir models/letters_baseline --video path/to/sign.mp4
python scripts/predict_webcam.py --model-dir models/letters_baseline --video path/to/sign.mp4 --headless
```

`--threshold`, `--stable-predictions`, and `--window-seconds` are adjustable.
Lowering the threshold makes predictions easier to display but does not improve
recognition accuracy. Confidence is an uncalibrated model score, and filtering
does not guarantee rejection of incorrect signs or non-sign movements.

## Dataset contract and current results

The inspected data has 1,598 sequences, 18 frames per sequence, 227 features,
five participants, and 24 letters. **H and J are absent and cannot be predicted.**
The feature layout is:

| Indices | Content |
| --- | --- |
| 0–98 | 33 pose landmarks, XYZ |
| 99–161 | 21 left-hand landmarks, XYZ |
| 162–224 | 21 right-hand landmarks, XYZ |
| 225–226 | Left/right hand presence flags |

Coordinates are image-normalized MediaPipe landmarks, not world landmarks.
For each frame, subtract the XYZ midpoint of pose shoulders 11/12 and divide
XYZ by their XY distance. Missing hands stay zero; presence flags are unchanged.
The shared implementation reproduces **every saved body/hand feature exactly**
from `processed_18_frames/sequences.npy[..., 1404:]`. Frame selection uses
uniform indices with floor rounding, also verified against every metadata row.
No padding mask is needed for these resampled sequences. The median clip span
is about 2.94 seconds, not 18 consecutive webcam frames.

The initial baseline (seed 42, default architecture, validation `rithika`, test
`mathur`) stopped after 26 epochs and selected epoch 11:

| Split | Sequences | Accuracy |
| --- | ---: | ---: |
| Validation | 508 | 15.35% |
| Test | 363 | 8.54% |

Training used 727 sequences. Only `angel` covers all 24 letters among the
remaining training participants; `aldebaran` covers A–E and `alix` only a few
letters. This is a runnable experimental baseline, **not a reliable learning
assessment model**. Gather more complete recordings across signers, check
capture consistency and per-letter errors, and select improvements using the
validation split. Do not repeatedly tune against the test score. Recognizing a
letter does not establish that the sign was performed correctly.

## Website letter exercise

`scripts/letter_predictor.py` exposes recorded-attempt inference for the sibling
`SignSenseGDG` website. Its Letter island uses the existing reference JPGs,
shuffles supported letters and sends timed, unmirrored JPEG camera frames to
the website's local `serve.py`. It reuses the same normalization, frame sampling
and `letters_three_signers` checkpoint as the webcam demo. Lost tracking produces
an uncertain result; only accepted matching predictions update local progress.

Run `python serve.py` from `C:\Users\mathu\SignSenseGDG` using the Python
environment with this repository's requirements installed. See that repository's
README for configuration and the localhost exercise URL. No model files need
to be copied to the frontend. This remains practice feedback with the existing
model limitations, not a validated technique assessment.

## Verification commands

```powershell
python -m unittest discover -s tests -v
```

Tests cover missing hands, degenerate shoulders, MediaPipe result conversion,
and exact normalization/frame-sampling parity with the local datasets. Dataset
checks skip when participant files are unavailable. Training, saved-model
inference and a headless blank-video MediaPipe smoke test can run without a
webcam. Actual live recognition must still be tested by a person signing.
