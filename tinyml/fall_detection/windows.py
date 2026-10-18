"""Turn recordings into fixed-length labelled windows."""

from __future__ import annotations

import numpy as np

from datasets import Recording


def make_windows(
    rec: Recording,
    window_len: int,
    adl_stride: int,
    fall_offsets: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut one recording into windows of shape (window_len, channels).

    ADL recordings: sliding windows with `adl_stride`, all labelled 0.
    Fall recordings: the impact is taken as the peak of |acc|; `fall_offsets`
    windows are placed so the peak lands between 30% and 80% of the window
    (some pre-impact free fall and some post-impact rest is always visible).
    Other windows from a fall trial are dropped because their label is
    ambiguous.
    """
    x = rec.data
    if len(x) < window_len:
        return np.empty((0, window_len, x.shape[1]), np.float32), np.empty((0,), np.int64)

    if not rec.is_fall:
        starts = range(0, len(x) - window_len + 1, adl_stride)
        w = np.stack([x[s : s + window_len] for s in starts])
        return w, np.zeros(len(w), np.int64)

    peak = int(np.argmax(np.linalg.norm(x[:, 0:3], axis=1)))
    pos = rng.uniform(0.3, 0.8, fall_offsets)
    starts = np.clip(peak - (pos * window_len).astype(int), 0, len(x) - window_len)
    w = np.stack([x[s : s + window_len] for s in np.unique(starts)])
    return w, np.ones(len(w), np.int64)


def random_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    """Small random 3D rotation, used to make the model tolerant of how the watch sits on the wrist."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(-max_deg, max_deg))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def augment(x: np.ndarray, rng: np.random.Generator, max_rot_deg: float = 30.0) -> np.ndarray:
    """Rotate acc and gyro together, jitter the scale and add sensor noise."""
    out = np.empty_like(x)
    for i in range(len(x)):
        r = random_rotation(rng, max_rot_deg).astype(np.float32)
        out[i, :, 0:3] = x[i, :, 0:3] @ r.T
        if x.shape[2] >= 6:
            out[i, :, 3:6] = x[i, :, 3:6] @ r.T
    out *= rng.uniform(0.9, 1.1, (len(x), 1, x.shape[2])).astype(np.float32)
    out += rng.normal(0, 0.01, out.shape).astype(np.float32)
    return out
