"""IMU recordings to replay: SisFall, MobiFall v2 and a synthetic generator.

All loaders produce :class:`Recording` objects with accelerometer data in g
and gyroscope data in deg/s, which is what the QMI8658 driver on the watch
reports. Loading is lazy so that thousands of files can be listed cheaply.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

G = 9.80665


@dataclass
class Recording:
    name: str
    activity: str          # dataset activity code, e.g. "F05" or "FOL"
    is_fall: bool
    subject: str
    source: str            # "sisfall", "mobifall" or "synthetic"
    fs: float
    acc: np.ndarray        # (N, 3) in g
    gyr: Optional[np.ndarray] = None  # (N, 3) in deg/s

    @property
    def duration_s(self) -> float:
        return len(self.acc) / self.fs


@dataclass
class RecordingRef:
    """Metadata plus a loader; the arrays are read only when needed."""
    name: str
    activity: str
    is_fall: bool
    subject: str
    source: str
    loader: Callable[[], Recording] = field(repr=False)

    def load(self) -> Recording:
        return self.loader()


def resample(rec: Recording, fs: float) -> Recording:
    """Linearly resample to ``fs`` Hz (the rate the virtual watch runs at)."""
    if abs(rec.fs - fs) < 1e-9:
        return rec
    n_in = len(rec.acc)
    t_in = np.arange(n_in) / rec.fs
    n_out = max(1, int(np.floor(t_in[-1] * fs)) + 1)
    t_out = np.arange(n_out) / fs

    def interp(a: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(t_out, t_in, a[:, i]) for i in range(3)], axis=1)

    return Recording(rec.name, rec.activity, rec.is_fall, rec.subject, rec.source,
                     fs, interp(rec.acc), None if rec.gyr is None else interp(rec.gyr))


# --------------------------------------------------------------------------
# SisFall (Sucerquia et al., 2017). 200 Hz, waist-mounted, 9 columns per row:
# ADXL345 xyz, ITG3200 xyz, MMA8451Q xyz as raw ADC counts. Files are named
# like SisFall_dataset/SA01/F05_SA01_R01.txt; F01-F15 are falls, D01-D19 ADLs.
# --------------------------------------------------------------------------

SISFALL_FS = 200.0
_SISFALL_ADXL345 = (2 * 16) / 2 ** 13   # +-16 g, 13 bit
_SISFALL_ITG3200 = (2 * 2000) / 2 ** 16  # +-2000 deg/s, 16 bit
_SISFALL_MMA8451Q = (2 * 8) / 2 ** 14    # +-8 g, 14 bit
_SISFALL_NAME = re.compile(r"^([DF]\d{2})_(S[AE]\d{2})_(R\d{2})\.txt$")


def parse_sisfall_file(path: Path, accel: str = "adxl345") -> Recording:
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            parts = [p for p in re.split(r"[,\s]+", line) if p]
            if len(parts) < 9:
                continue
            rows.append([float(p) for p in parts[:9]])
    if not rows:
        raise ValueError(f"no samples in {path}")
    raw = np.asarray(rows)
    if accel == "adxl345":
        acc = raw[:, 0:3] * _SISFALL_ADXL345
    elif accel == "mma8451q":
        acc = raw[:, 6:9] * _SISFALL_MMA8451Q
    else:
        raise ValueError(f"unknown SisFall accelerometer {accel!r}")
    gyr = raw[:, 3:6] * _SISFALL_ITG3200
    m = _SISFALL_NAME.match(path.name)
    activity, subject = (m.group(1), m.group(2)) if m else ("?", "?")
    return Recording(path.stem, activity, activity.startswith("F"), subject,
                     "sisfall", SISFALL_FS, acc, gyr)


def list_sisfall(root: str | Path, accel: str = "adxl345") -> list[RecordingRef]:
    refs = []
    for path in sorted(Path(root).rglob("*.txt")):
        m = _SISFALL_NAME.match(path.name)
        if not m:
            continue
        activity, subject = m.group(1), m.group(2)
        refs.append(RecordingRef(path.stem, activity, activity.startswith("F"),
                                 subject, "sisfall",
                                 functools.partial(parse_sisfall_file, path, accel)))
    return refs


# --------------------------------------------------------------------------
# MobiFall v2.0 (Vavoulas et al., 2014). Smartphone in a trouser pocket,
# ~87 Hz with irregular timestamps. Layout:
#   sub<N>/{ADL,FALLS}/<CODE>/<CODE>_{acc,gyro,ori}_<N>_<trial>.txt
# Each file has a '#' header, an '@DATA' line, then "timestamp_ns,x,y,z".
# acc is in m/s^2, gyro in rad/s. Falls: FOL, FKL, BSC, SDL.
# --------------------------------------------------------------------------

MOBIFALL_FALLS = {"FOL", "FKL", "BSC", "SDL"}
_MOBIFALL_NAME = re.compile(r"^([A-Z]{3})_acc_(\d+)_(\d+)\.txt$")


def _read_mobifall_table(path: Path) -> np.ndarray:
    rows = []
    in_data = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not in_data:
                if line.upper().startswith("@DATA"):
                    in_data = True
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                rows.append([float(p) for p in parts[:4]])
    if not rows:
        raise ValueError(f"no samples in {path}")
    return np.asarray(rows)


def parse_mobifall_file(acc_path: Path, fs: float = 50.0) -> Recording:
    """Read an acc file (and its gyro sibling) onto a uniform ``fs`` grid."""
    acc_raw = _read_mobifall_table(acc_path)
    acc_raw = acc_raw[np.argsort(acc_raw[:, 0], kind="stable")]
    t = (acc_raw[:, 0] - acc_raw[0, 0]) / 1e9
    t_out = np.arange(0.0, t[-1], 1.0 / fs) if t[-1] > 0 else np.zeros(1)

    def grid(tt: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(t_out, tt, xyz[:, i]) for i in range(3)], axis=1)

    acc = grid(t, acc_raw[:, 1:4]) / G
    gyr = None
    gyro_path = acc_path.with_name(acc_path.name.replace("_acc_", "_gyro_"))
    if gyro_path.exists():
        g_raw = _read_mobifall_table(gyro_path)
        g_raw = g_raw[np.argsort(g_raw[:, 0], kind="stable")]
        tg = (g_raw[:, 0] - acc_raw[0, 0]) / 1e9
        gyr = np.degrees(grid(tg, g_raw[:, 1:4]))
    m = _MOBIFALL_NAME.match(acc_path.name)
    activity, subject = (m.group(1), f"sub{m.group(2)}") if m else ("?", "?")
    return Recording(acc_path.stem.replace("_acc", ""), activity,
                     activity in MOBIFALL_FALLS, subject, "mobifall", fs, acc, gyr)


def list_mobifall(root: str | Path, fs: float = 50.0) -> list[RecordingRef]:
    refs = []
    for path in sorted(Path(root).rglob("*_acc_*.txt")):
        m = _MOBIFALL_NAME.match(path.name)
        if not m:
            continue
        activity = m.group(1)
        refs.append(RecordingRef(path.stem.replace("_acc", ""), activity,
                                 activity in MOBIFALL_FALLS, f"sub{m.group(2)}",
                                 "mobifall",
                                 functools.partial(parse_mobifall_file, path, fs)))
    return refs


# --------------------------------------------------------------------------
# Synthetic recordings, so the simulator runs without any dataset download.
# They are crude on purpose: good for exercising the pipeline, not for
# measuring a model.
# --------------------------------------------------------------------------

SYNTHETIC_ADLS = ("STD", "WAL", "JOG", "SIT", "JMP")


def synthetic_recording(kind: str, rng: np.random.Generator, fs: float = 50.0,
                        duration_s: float = 10.0, name: str = "syn") -> Recording:
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    acc = np.zeros((n, 3))
    acc[:, 2] = 1.0  # gravity on z while upright
    gyr = rng.normal(0, 2.0, (n, 3))
    # walking and jogging ramp up and wind down instead of stopping dead
    env = np.clip(np.minimum(t, t[-1] - t) / 3.0, 0.0, 1.0)
    if kind == "WAL":
        f = rng.uniform(1.6, 2.1)
        acc[:, 2] += env * 0.25 * np.sin(2 * np.pi * f * t)
        acc[:, 0] += env * 0.15 * np.sin(np.pi * f * t)
        gyr[:, 1] += env * 30 * np.sin(np.pi * f * t)
    elif kind == "JOG":
        f = rng.uniform(2.5, 3.0)
        acc[:, 2] += env * (2.6 * np.abs(np.sin(np.pi * f * t)) - 0.8)
        acc[:, 0] += env * 0.3 * np.sin(np.pi * f * t)
        gyr[:, 1] += env * 80 * np.sin(np.pi * f * t)
    elif kind == "SIT":
        # sit down quickly: a soft bump, then tilted posture
        i = int(n * rng.uniform(0.3, 0.5))
        w = int(0.4 * fs)
        acc[i:i + w, 2] += 0.7 * np.hanning(w)[: len(acc[i:i + w])]
        tilt = np.radians(rng.uniform(20, 40))
        acc[i + w:, 2] = np.cos(tilt)
        acc[i + w:, 0] = np.sin(tilt)
    elif kind == "JMP":
        # jump and land, then stand still: a hard negative for impact+stillness
        i = int(n * rng.uniform(0.3, 0.5))
        air = int(0.3 * fs)
        acc[i:i + air, :] *= 0.1
        land = max(2, int(0.08 * fs))
        acc[i + air:i + air + land, 2] = rng.uniform(2.5, 4.0) * np.hanning(land + 2)[1:-1]
    elif kind == "FALL":
        i = int(n * rng.uniform(0.3, 0.45))
        ff = int(rng.uniform(0.25, 0.45) * fs)       # free fall
        acc[i:i + ff, :] *= rng.uniform(0.1, 0.4)
        j = i + ff
        imp = max(2, int(0.08 * fs))                  # impact
        peak = rng.uniform(3.0, 6.0)
        axis_dir = rng.normal(size=3)
        axis_dir /= np.linalg.norm(axis_dir)
        acc[j:j + imp, :] = axis_dir * peak * np.hanning(imp + 2)[1:-1, None]
        gyr[i:j + imp, :] += rng.normal(0, 150, (j + imp - i, 3))
        # lying still, gravity now on a horizontal axis
        k = j + imp
        lie = np.array([rng.choice([-1, 1]) * 1.0, 0.0, 0.0])
        acc[k:, :] = lie
    elif kind != "STD":
        raise ValueError(f"unknown synthetic kind {kind!r}")
    acc += rng.normal(0, 0.02, acc.shape)
    activity = "FALL" if kind == "FALL" else kind
    return Recording(name, activity, kind == "FALL", "synthetic", "synthetic", fs, acc, gyr)


def list_synthetic(count: int = 200, fall_ratio: float = 0.3, fs: float = 50.0,
                   seed: int = 0) -> list[RecordingRef]:
    rng = np.random.default_rng(seed)
    refs = []
    for i in range(count):
        kind = "FALL" if rng.random() < fall_ratio else str(rng.choice(SYNTHETIC_ADLS))
        name = f"syn_{kind}_{i:04d}"
        rec_seed = int(rng.integers(2 ** 31))
        refs.append(RecordingRef(
            name, kind, kind == "FALL", "synthetic", "synthetic",
            functools.partial(_synthetic_from_seed, kind, rec_seed, fs, name)))
    return refs


def _synthetic_from_seed(kind: str, seed: int, fs: float, name: str) -> Recording:
    return synthetic_recording(kind, np.random.default_rng(seed), fs, name=name)


def still_segment(fs: float, duration_s: float, rng: np.random.Generator,
                  gravity: Optional[np.ndarray] = None) -> np.ndarray:
    """Nearly still at 1 g; used as a gap between replayed recordings.

    ``gravity`` is the direction to hold (e.g. the last sample of the previous
    recording) so the gap does not add an artificial jump.
    """
    n = max(1, int(duration_s * fs))
    g = np.array([0.0, 0.0, 1.0]) if gravity is None else np.asarray(gravity, float)
    norm = np.linalg.norm(g)
    g = np.array([0.0, 0.0, 1.0]) if norm < 1e-6 else g / norm
    return np.tile(g, (n, 1)) + rng.normal(0, 0.01, (n, 3))
