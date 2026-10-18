"""Loaders for public fall datasets.

Every loader yields Recording objects with the same layout so the rest of the
pipeline does not care where the data came from:

    data:  float32 array (T, 6) = [ax, ay, az] in g, [gx, gy, gz] in deg/s
    fs:    sampling rate of `data` in Hz (uniform)
    is_fall: True for fall trials, False for activities of daily living (ADL)
    subject: dataset-prefixed subject id, used for subject-wise splits

Units match what the QMI8658 on the watch gives after dividing m/s^2 by g.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

G = 9.80665


@dataclass
class Recording:
    data: np.ndarray
    fs: float
    is_fall: bool
    subject: str
    activity: str
    source: str


def resample_uniform(t: np.ndarray, x: np.ndarray, fs_out: float) -> np.ndarray:
    """Resample (possibly irregular) samples x(t) onto a uniform fs_out grid.

    A moving average over one output period is applied first as a cheap
    anti-alias filter, so downsampling 200 Hz -> 50 Hz does not fold
    high-frequency vibration into the band the model sees.
    """
    t = np.asarray(t, dtype=np.float64)
    order = np.argsort(t)
    t, x = t[order], x[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, x = t[keep], x[keep]

    fs_in = (len(t) - 1) / (t[-1] - t[0])
    k = int(round(fs_in / fs_out))
    if k > 1:
        kernel = np.ones(k) / k
        x = np.stack([np.convolve(x[:, c], kernel, mode="same") for c in range(x.shape[1])], axis=1)

    grid = np.arange(t[0], t[-1], 1.0 / fs_out)
    return np.stack([np.interp(grid, t, x[:, c]) for c in range(x.shape[1])], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# SisFall (Sucerquia et al., 2017)
# Layout: SisFall_dataset/<SA01|SE01..>/<D01..D19|F01..F15>_<subject>_R<nn>.txt
# Each line: 9 comma-separated raw ints, terminated by ';'
#   cols 0-2 ADXL345 (+-16 g, 13 bit), cols 3-5 ITG3200 (+-2000 deg/s, 16 bit),
#   cols 6-8 MMA8451Q (unused here). Sampled at 200 Hz.
# --------------------------------------------------------------------------
SISFALL_FS = 200.0
_ADXL345_TO_G = (2 * 16) / 2**13
_ITG3200_TO_DPS = (2 * 2000) / 2**16
_SISFALL_NAME = re.compile(r"^([DF]\d{2})_(S[AE]\d{2})_R\d{2}\.txt$")


def load_sisfall(root: Path, fs_out: float) -> Iterator[Recording]:
    root = Path(root)
    files = sorted(root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No SisFall .txt files under {root}")
    for f in files:
        m = _SISFALL_NAME.match(f.name)
        if not m:
            continue
        activity, subject = m.groups()
        rows = []
        for line in f.read_text(errors="ignore").splitlines():
            line = line.strip().rstrip(";")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                rows.append([float(v) for v in parts[:6]])
            except ValueError:
                continue
        if len(rows) < SISFALL_FS:
            continue
        raw = np.asarray(rows, dtype=np.float64)
        data = np.empty_like(raw)
        data[:, 0:3] = raw[:, 0:3] * _ADXL345_TO_G
        data[:, 3:6] = raw[:, 3:6] * _ITG3200_TO_DPS
        t = np.arange(len(data)) / SISFALL_FS
        yield Recording(
            data=resample_uniform(t, data, fs_out),
            fs=fs_out,
            is_fall=activity.startswith("F"),
            subject=f"sisfall_{subject}",
            activity=activity,
            source="sisfall",
        )


# --------------------------------------------------------------------------
# MobiFall v2.0 (Vavoulas et al., 2014)
# Layout: MobiFall_Dataset_v2.0/sub<N>/<ADL|FALLS>/<CODE>/<CODE>_<acc|gyro|ori>_<N>_<trial>.txt
# Header lines start with '#' or '@'; data lines are "timestamp_ns, x, y, z".
# acc in m/s^2, gyro in rad/s, irregular timing (~87 Hz acc, ~200 Hz gyro).
# --------------------------------------------------------------------------
_MOBIFALL_NAME = re.compile(r"^([A-Z]{3})_acc_(\d+)_(\d+)\.txt$")


def _read_mobifall_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line[0] in "#@":
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            rows.append([float(v) for v in parts[:4]])
        except ValueError:
            continue
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0] * 1e-9, arr[:, 1:4]


def load_mobifall(root: Path, fs_out: float) -> Iterator[Recording]:
    root = Path(root)
    acc_files = sorted(root.rglob("*_acc_*.txt"))
    if not acc_files:
        raise FileNotFoundError(f"No MobiFall *_acc_*.txt files under {root}")
    for acc_path in acc_files:
        m = _MOBIFALL_NAME.match(acc_path.name)
        if not m:
            continue
        activity, subject, _ = m.groups()
        gyro_path = acc_path.with_name(acc_path.name.replace("_acc_", "_gyro_"))
        if not gyro_path.exists():
            continue
        ta, acc = _read_mobifall_txt(acc_path)
        tg, gyro = _read_mobifall_txt(gyro_path)
        if len(ta) < 50 or len(tg) < 50:
            continue
        # Align both streams on their common time span before resampling.
        t0, t1 = max(ta[0], tg[0]), min(ta[-1], tg[-1])
        if t1 - t0 < 2.0:
            continue
        acc_u = resample_uniform(ta, acc / G, fs_out)
        gyro_u = resample_uniform(tg, np.degrees(gyro), fs_out)
        grid_a = ta[0] + np.arange(len(acc_u)) / fs_out
        grid_g = tg[0] + np.arange(len(gyro_u)) / fs_out
        grid = np.arange(t0, t1, 1.0 / fs_out)
        data = np.concatenate(
            [
                np.stack([np.interp(grid, grid_a, acc_u[:, c]) for c in range(3)], axis=1),
                np.stack([np.interp(grid, grid_g, gyro_u[:, c]) for c in range(3)], axis=1),
            ],
            axis=1,
        ).astype(np.float32)
        is_fall = "FALLS" in (p.upper() for p in acc_path.parts)
        yield Recording(
            data=data,
            fs=fs_out,
            is_fall=is_fall,
            subject=f"mobifall_sub{subject}",
            activity=activity,
            source="mobifall",
        )


# --------------------------------------------------------------------------
# Synthetic recordings: a stand-in for real data so the pipeline can be
# smoke-tested (and CI-run) without downloading anything. Not for real use.
# --------------------------------------------------------------------------
def load_synthetic(n_subjects: int, fs_out: float, seed: int = 0) -> Iterator[Recording]:
    rng = np.random.default_rng(seed)
    duration = 12.0
    n = int(duration * fs_out)
    t = np.arange(n) / fs_out
    for s in range(n_subjects):
        for trial in range(30):
            is_fall = trial % 3 == 0
            acc = np.zeros((n, 3))
            acc[:, 2] = 1.0
            gyro = rng.normal(0, 3, (n, 3))
            step_f = rng.uniform(1.4, 2.6)
            amp = rng.uniform(0.0, 0.6)
            acc[:, 2] += amp * np.sin(2 * np.pi * step_f * t)
            acc[:, 0] += 0.3 * amp * np.sin(2 * np.pi * step_f * t / 2)
            if not is_fall and rng.random() < 0.3:
                # Sit down hard / jump: a short spike without the free-fall + lying pattern.
                k = rng.integers(n // 4, 3 * n // 4)
                acc[k : k + 5, 2] += rng.uniform(1.0, 2.0)
            if is_fall:
                k = rng.integers(n // 4, 3 * n // 4)
                ff = int(0.4 * fs_out)
                acc[k - ff : k] *= np.linspace(1, 0.15, ff)[:, None]
                acc[k : k + 4] += rng.normal(0, 1, 3) + np.array([0, 0, rng.uniform(2.5, 5)])
                gyro[k - ff : k + 10] += rng.normal(0, 150, 3)
                # Lying still afterwards: gravity on the x axis.
                acc[k + 4 :] = np.array([1.0, 0.0, 0.1]) + rng.normal(0, 0.02, (n - k - 4, 3))
            acc += rng.normal(0, 0.03, acc.shape)
            yield Recording(
                data=np.concatenate([acc, gyro], axis=1).astype(np.float32),
                fs=fs_out,
                is_fall=is_fall,
                subject=f"synthetic_{s:02d}",
                activity="FALL" if is_fall else "ADL",
                source="synthetic",
            )
