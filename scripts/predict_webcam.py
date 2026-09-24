#!/usr/bin/env python3
"""Recognize letters using MediaPipe Holistic and an exported Signsense LSTM."""
import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np

from sign_features import MODEL_DIMS, model_features, features_from_result, sample_window


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--landmarker', type=Path, default=Path('models/holistic_landmarker.task'))
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--video', type=Path, help='Use a recorded video instead of a webcam')
    p.add_argument('--headless', action='store_true', help='Print predictions without a window (video only)')
    p.add_argument('--threshold', type=float, default=0.8)
    p.add_argument('--stable-predictions', type=int, default=3)
    p.add_argument('--prediction-interval', type=float, default=0.25)
    p.add_argument('--window-seconds', type=float, help='Override training median clip duration')
    p.add_argument('--mirror-input', action='store_true', help='Flip BEFORE extraction if recording used mirrored input')
    p.add_argument('--max-frames', type=int, help='Stop after this many frames (for smoke tests)')
    args = p.parse_args()
    if not 0 <= args.threshold <= 1 or args.stable_predictions < 1 or args.prediction_interval <= 0:
        p.error('Invalid confidence, stability, or prediction interval')
    if args.headless and not args.video:
        p.error('--headless requires --video')
    if args.window_seconds is not None and args.window_seconds <= 0:
        p.error('--window-seconds must be positive')
    if args.max_frames is not None and args.max_frames < 1:
        p.error('--max-frames must be positive')
    return args


def main():
    args = parse_args()
    import cv2
    import mediapipe as mp
    import tensorflow as tf

    config = json.loads((args.model_dir / 'inference_config.json').read_text())
    if config['schema'] not in MODEL_DIMS or config['feature_dim'] != MODEL_DIMS[config['schema']]:
        raise SystemExit('Unsupported feature schema; retrain with the current trainer')
    if not args.landmarker.is_file():
        raise SystemExit('Missing MediaPipe model. Run: python scripts/download_landmarker.py')
    model = tf.keras.models.load_model(args.model_dir / 'best_lstm_model.keras', compile=False)
    steps, dims = config['timesteps'], config['feature_dim']
    names = config['class_names']
    if tuple(model.input_shape[1:]) != (steps, dims) or model.output_shape[-1] != len(names):
        raise SystemExit('Model and inference_config.json disagree')
    window_seconds = args.window_seconds or config['window_seconds']
    cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not cap.isOpened():
        cap.release()
        raise SystemExit('Could not open video/camera; check its path/index and camera permissions')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.video and (not np.isfinite(fps) or fps <= 0):
        cap.release()
        raise SystemExit('Video has no usable frame rate')
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.landmarker)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO)
    frames = deque()
    recent = deque(maxlen=args.stable_predictions)
    status = 'Show a sign with shoulders visible'
    detail = ''
    last_prediction = -float('inf')
    last_ms = -1
    started = time.monotonic()
    frame_index = 0
    try:
        with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = frame_index / fps if args.video else time.monotonic() - started
                frame_index += 1
                if args.mirror_input:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(last_ms + 1, int(now * 1000))
                last_ms = timestamp_ms
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                features = features_from_result(result)
                # Do not bridge detection failures or long capture pauses with stale frames.
                if features is None or (frames and now - frames[-1][0] > 0.5):
                    frames.clear()
                    recent.clear()
                    status = 'Show a sign with shoulders visible'
                    detail = ''
                if features is not None:
                    frames.append((now, features))
                    # Retain the frame immediately before the window boundary.
                    while len(frames) > 1 and frames[1][0] <= now - window_seconds:
                        frames.popleft()
                    elapsed = now - frames[0][0]
                    if elapsed < window_seconds or len(frames) < steps:
                        status = f'Collecting sign: {min(elapsed / window_seconds, 1):.0%}'
                    elif now - last_prediction >= args.prediction_interval:
                        sequence = sample_window([f for _, f in frames], steps)
                        sequence = model_features(sequence, config['schema'])
                        probs = model(sequence[None], training=False).numpy()[0]
                        index = int(probs.argmax())
                        confidence = float(probs[index])
                        recent.append(index if confidence >= args.threshold else -1)
                        stable = len(recent) == recent.maxlen and all(i == index for i in recent)
                        if stable:
                            status = f'Letter: {names[index]} ({confidence:.0%})'
                        elif confidence < args.threshold:
                            status = f'Uncertain: best guess {names[index]} ({confidence:.0%} < {args.threshold:.0%})'
                        else:
                            status = f'Uncertain: best guess {names[index]} ({confidence:.0%}), waiting for agreement'
                        top = np.argsort(probs)[-3:][::-1]
                        detail = 'Top guesses: ' + ' | '.join(f'{names[i]} {probs[i]:.0%}' for i in top)
                        print(f'{now:7.2f}s  {status} | {detail}', flush=True)
                        last_prediction = now
                if not args.headless:
                    # Selfie preview only: extraction above uses the original image unless requested.
                    display = cv2.flip(frame, 1)
                    height, width = display.shape[:2]
                    for points in (result.left_hand_landmarks, result.right_hand_landmarks):
                        for point in points:
                            cv2.circle(display, (int((1 - point.x) * width), int(point.y * height)), 3, (0, 255, 0), -1)
                    cv2.rectangle(display, (0, 0), (width, 110), (25, 25, 25), -1)
                    text_width = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                    font_scale = 0.6 * min(1, max(width - 24, 1) / max(text_width, 1))
                    cv2.putText(display, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
                    cv2.putText(display, detail, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                    cv2.putText(display, 'Q: quit | R: reset | hold each sign for about 3 seconds', (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                    cv2.imshow('Signsense - Auslan letters', display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                    if key == ord('r'):
                        frames.clear()
                        recent.clear()
                        status = 'Show a sign with shoulders visible'
                        detail = ''
                if args.max_frames and frame_index >= args.max_frames:
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
