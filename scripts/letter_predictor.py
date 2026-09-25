"""Shared, recorded-attempt inference for the local SignSense letter exercise."""
import base64
import binascii
import json
from pathlib import Path

import numpy as np

from sign_features import MODEL_DIMS, features_from_result, model_features, sample_window


class InvalidAttempt(ValueError):
    """The capture cannot safely be processed."""


def validate_frames(frames, config):
    if not isinstance(frames, list) or not config['timesteps'] <= len(frames) <= 90:
        raise InvalidAttempt('Capture at least 18 frames and no more than 90.')
    times = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise InvalidAttempt('Invalid camera frame.')
        timestamp = frame.get('timestamp_ms')
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp):
            raise InvalidAttempt('Invalid frame timestamp.')
        if timestamp < 0 or timestamp > 10000 or (times and timestamp - times[-1] < 1):
            raise InvalidAttempt('Frame timestamps must increase within a ten-second capture.')
        if times and timestamp - times[-1] > 500:
            raise InvalidAttempt('Camera paused during capture. Please try again.')
        encoded = frame.get('jpeg')
        if not isinstance(encoded, str) or not 1 <= len(encoded) <= 350000:
            raise InvalidAttempt('Invalid camera image size.')
        times.append(timestamp)
    duration = (times[-1] - times[0]) / 1000
    if not config['window_seconds'] <= duration <= config['window_seconds'] + 1:
        raise InvalidAttempt('Capture was too short or too long. Please try again.')
    return times


def verdict(probabilities, names, expected, threshold=0.8):
    index = int(np.argmax(probabilities))
    detected, score = names[index], float(probabilities[index])
    if score < threshold:
        status, feedback = 'uncertain', 'We could not recognise that clearly. Try again.'
    elif detected == expected:
        status, feedback = 'match', f'Correct! We recognised {expected}.'
    else:
        status, feedback = 'different_sign', f'We detected {detected}, rather than {expected}. Look at the photo and try again.'
    return dict(status=status, detected_letter=detected, model_score=score,
                reason_code='low_score' if status == 'uncertain' else status, feedback=feedback)


class LetterPredictor:
    def __init__(self, model_dir, landmarker, threshold=0.8):
        import cv2
        import mediapipe as mp
        import tensorflow as tf
        self.cv2, self.mp = cv2, mp
        self.config = json.loads((Path(model_dir) / 'inference_config.json').read_text())
        cfg = self.config
        if MODEL_DIMS.get(cfg['schema']) != cfg['feature_dim']:
            raise ValueError('Unsupported feature schema.')
        self.model = tf.keras.models.load_model(Path(model_dir) / 'best_lstm_model.keras', compile=False)
        if tuple(self.model.input_shape[1:]) != (cfg['timesteps'], cfg['feature_dim']) or self.model.output_shape[-1] != len(cfg['class_names']):
            raise ValueError('Model and inference configuration disagree.')
        self.landmarker = str(landmarker)
        if not Path(landmarker).is_file():
            raise FileNotFoundError('Missing holistic_landmarker.task. Run scripts/download_landmarker.py.')
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold must be between zero and one.')
        self.threshold = threshold

    def predict(self, frames, expected):
        cfg, cv2, mp = self.config, self.cv2, self.mp
        if expected not in cfg['class_names']:
            raise InvalidAttempt('Unsupported letter.')
        times = validate_frames(frames, cfg)
        options = mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.landmarker),
            running_mode=mp.tasks.vision.RunningMode.VIDEO)
        features = []
        # Each attempt owns its tracker. Never carry landmarks between learners.
        with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as detector:
            for item, timestamp in zip(frames, times):
                try:
                    raw = base64.b64decode(item['jpeg'], validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise InvalidAttempt('Invalid JPEG encoding.') from exc
                # Bound decoded dimensions before asking OpenCV to allocate pixels.
                if not raw.startswith(b'\xff\xd8'):
                    raise InvalidAttempt('Expected a JPEG camera frame.')
                from PIL import Image
                from io import BytesIO
                try:
                    with Image.open(BytesIO(raw)) as header:
                        if header.format != 'JPEG' or not (160 <= header.width <= 960 and 120 <= header.height <= 720):
                            raise InvalidAttempt('Camera frames must be between 160x120 and 960x720.')
                except (OSError, Image.DecompressionBombError) as exc:
                    raise InvalidAttempt('Camera image could not be read.') from exc
                frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise InvalidAttempt('Camera image could not be decoded.')
                if cfg.get('mirror_input', False):
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(timestamp))
                value = features_from_result(result)
                if value is None:
                    return dict(status='uncertain', detected_letter=None, model_score=None,
                                reason_code='tracking_lost', feedback='Keep your shoulders and signing hands visible for the whole attempt, then try again.')
                features.append(value)
        sequence = model_features(sample_window(features, cfg['timesteps']), cfg['schema'])
        probs = self.model(sequence[None], training=False).numpy()[0]
        return verdict(probs, cfg['class_names'], expected, self.threshold)
