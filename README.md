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

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests cover missing hands, degenerate shoulders, MediaPipe result conversion,
and exact normalization/frame-sampling parity with the local datasets. Dataset
checks skip when participant files are unavailable. Training, saved-model
inference and a headless blank-video MediaPipe smoke test can run without a
webcam. Actual live recognition must still be tested by a person signing.
