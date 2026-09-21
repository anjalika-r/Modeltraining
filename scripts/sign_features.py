"""Shared feature contract, verified against processed_18_frames raw arrays."""
import json
from pathlib import Path

import numpy as np

FEATURE_DIM = 227
SCHEMA = 'pose33_left21_right21_xyz_shoulder_v1'


def normalize_body_hands(raw):
    """Accept (..., 227): pose XYZ, left XYZ, right XYZ, two presence flags."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-1] != FEATURE_DIM or not np.isfinite(raw).all():
        raise ValueError('Expected finite 227-feature body/hand input')
    pose = raw[..., :99].reshape(*raw.shape[:-1], 33, 3)
    origin = (pose[..., 11, :] + pose[..., 12, :]) / 2
    scale = np.linalg.norm(pose[..., 11, :2] - pose[..., 12, :2], axis=-1)
    if np.any(scale < 1e-6):
        raise ValueError('Shoulders missing or too close to normalize')
    result = raw.copy()
    for start, end, flag in [(0, 99, None), (99, 162, 225), (162, 225, 226)]:
        points = raw[..., start:end].reshape(*raw.shape[:-1], -1, 3)
        points = (points - origin[..., None, :]) / scale[..., None, None]
        if flag is not None:
            points = np.where(raw[..., flag, None, None] > 0, points, 0)
        result[..., start:end] = points.reshape(*raw.shape[:-1], end - start)
    return result


def features_from_result(result):
    """Use image-normalized landmarks, NOT MediaPipe world landmarks."""
    pose = result.pose_landmarks
    if len(pose) != 33:
        return None
    if any((pose[i].visibility or 0) < 0.5 for i in (11, 12)):
        return None
    raw = np.zeros(FEATURE_DIM, dtype=np.float32)
    for points, start, count, flag in [
        (pose, 0, 33, None),
        (result.left_hand_landmarks, 99, 21, 225),
        (result.right_hand_landmarks, 162, 21, 226),
    ]:
        if len(points) == count:
            raw[start:start + count * 3] = np.asarray(
                [(p.x, p.y, p.z) for p in points], dtype=np.float32).ravel()
            if flag is not None:
                raw[flag] = 1
    if not raw[225:].any():
        return None
    try:
        return normalize_body_hands(raw)
    except ValueError:
        return None


def load_class_names(data_dir, y):
    mapping = json.loads((Path(data_dir) / 'class_mapping.json').read_text())
    names = mapping['id_to_class']
    if not names or set(names) != {str(i) for i in range(len(names))}:
        raise ValueError('Class mapping must contain contiguous IDs starting at zero')
    ordered = [names[str(i)] for i in range(len(names))]
    if mapping.get('class_to_id') != {name: i for i, name in enumerate(ordered)}:
        raise ValueError('Class mapping directions disagree')
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('labels.npy must be a one-dimensional integer array')
    if not np.array_equal(np.unique(y), np.arange(len(ordered))):
        raise ValueError('Labels do not cover exactly the class mapping')
    return ordered


def sample_window(frames, timesteps):
    """Match dataset frame-index subsampling (linspace, floor to integer)."""
    if len(frames) < timesteps:
        raise ValueError('Not enough frames')
    return np.asarray(frames, dtype=np.float32)[
        np.linspace(0, len(frames) - 1, timesteps).astype(int)]
