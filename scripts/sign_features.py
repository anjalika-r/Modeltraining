"""Shared feature contract, verified against processed_18_frames raw arrays."""
import json
from pathlib import Path

import numpy as np

FEATURE_DIM = 227
SCHEMA = 'pose33_left21_right21_xyz_shoulder_v1'
ENGINEERED_SCHEMA = 'body_hands_shape_cross_xy_v2'
MODEL_SCHEMAS = {'baseline': SCHEMA, 'shape-contact': ENGINEERED_SCHEMA}
MODEL_DIMS = {SCHEMA: 227, ENGINEERED_SCHEMA: 497}
# Wrist, fingertips, and four finger bases. Cross distances are left-major.
CROSS_POINTS = [0, 4, 8, 12, 16, 20, 5, 9, 13, 17]


def model_features(body_hands, schema=ENGINEERED_SCHEMA):
    """Transform shoulder-normalized capture features, identically offline/live.

    v2 layout: original 227; left/right 83 each (63 wrist-centred XYZ,
    15 angles/pi, 5 extension ratios); 100 cross-hand XY distances;
    2 right-minus-left wrist XY; 2 geometry-valid flags.
    Invalid hand geometry is zeroed, including its retained coordinates.
    Validity detects degenerate geometry, NOT plausible tracking mistakes.
    """
    x = np.asarray(body_hands, dtype=np.float32)
    if x.ndim < 1 or x.shape[-1] != FEATURE_DIM or not np.isfinite(x).all():
        raise ValueError('Expected finite shoulder-normalized 227-feature input')
    if schema == SCHEMA:
        return x
    if schema != ENGINEERED_SCHEMA:
        raise ValueError(f'Unsupported feature schema: {schema}')
    base = x.copy()
    hands, valid, blocks = [], [], []
    chains = np.arange(1, 21).reshape(5, 4)
    vertices = chains[:, :3].reshape(-1)
    proximal = np.where(vertices % 4 == 1, 0, vertices - 1)
    eps = 1e-6
    for start, flag in [(99, 225), (162, 226)]:
        h = x[..., start:start + 63].reshape(*x.shape[:-1], 21, 3)
        palm = np.linalg.norm(h[..., 5, :] - h[..., 17, :], axis=-1)
        a = h[..., proximal, :] - h[..., vertices, :]
        b = h[..., vertices + 1, :] - h[..., vertices, :]
        an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        ok = (x[..., flag] > 0) & (palm > eps) & (an > eps).all(axis=-1) & (bn > eps).all(axis=-1)
        local = (h - h[..., :1, :]) / np.maximum(palm, eps)[..., None, None]
        cosine = np.sum(a * b, axis=-1) / np.maximum(an * bn, eps * eps)
        angles = np.arccos(np.clip(cosine, -1, 1)) / np.pi
        fingers = h[..., chains, :]
        length = np.linalg.norm(np.diff(fingers, axis=-2), axis=-1).sum(axis=-1)
        ratios = np.linalg.norm(fingers[..., -1, :] - fingers[..., 0, :], axis=-1) / np.maximum(length, eps)
        block = np.concatenate([local.reshape(*x.shape[:-1], 63), angles, np.clip(ratios, 0, 1)], axis=-1)
        blocks.append(np.where(ok[..., None], block, 0))
        base[..., start:start + 63] = np.where(ok[..., None], base[..., start:start + 63], 0)
        hands.append(h)
        valid.append(ok)
    both = valid[0] & valid[1]
    delta = hands[0][..., CROSS_POINTS, :2][..., :, None, :] - hands[1][..., CROSS_POINTS, :2][..., None, :, :]
    distances = np.linalg.norm(delta, axis=-1).reshape(*x.shape[:-1], 100)
    wrist = hands[1][..., 0, :2] - hands[0][..., 0, :2]
    return np.concatenate([base, *blocks, np.where(both[..., None], distances, 0),
                           np.where(both[..., None], wrist, 0), np.stack(valid, axis=-1)], axis=-1).astype(np.float32)


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
