"""On-watch fall detectors for virtual clients.

A detector is fed one accelerometer sample (g) at a time and returns a
:class:`Detection` when it decides a fall happened. The threshold detector is
a baseline that runs today; the TinyML model from the training pipeline can be
plugged in later by implementing the same two members (``name``, ``update``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class Detection:
    sample_index: int   # index of the sample that triggered the decision
    confidence: float
    peak_g: float


class Detector(Protocol):
    name: str

    def update(self, acc: np.ndarray, gyr: Optional[np.ndarray] = None) -> Optional[Detection]:
        ...


class ThresholdDetector:
    """Impact followed by stillness.

    1. The acceleration magnitude exceeds ``impact_g``.
    2. ``settle_s`` seconds later, the magnitude over the last ``still_s``
       seconds has a standard deviation below ``still_std_g``.
    Afterwards the detector ignores new impacts for ``refractory_s``.
    """

    name = "threshold-v1"

    def __init__(self, fs: float, impact_g: float = 2.5, settle_s: float = 1.5,
                 still_s: float = 1.0, still_std_g: float = 0.15,
                 refractory_s: float = 5.0):
        self.fs = fs
        self.impact_g = impact_g
        self.settle = int(settle_s * fs)
        self.still_n = max(2, int(still_s * fs))
        self.still_std_g = still_std_g
        self.refractory = int(refractory_s * fs)
        self._mags: deque[float] = deque(maxlen=self.still_n)
        self._i = -1
        self._impact_at: Optional[int] = None
        self._peak = 0.0
        self._quiet_until = -1

    def update(self, acc: np.ndarray, gyr: Optional[np.ndarray] = None) -> Optional[Detection]:
        self._i += 1
        mag = float(np.linalg.norm(acc))
        self._mags.append(mag)
        if self._i < self._quiet_until:
            return None
        if mag >= self.impact_g:
            self._peak = max(self._peak, mag)
            self._impact_at = self._i
            return None
        if self._impact_at is None or self._i - self._impact_at < self.settle:
            return None
        std = float(np.std(self._mags))
        peak = self._peak
        self._impact_at = None
        self._peak = 0.0
        if std >= self.still_std_g:
            return None  # moved on after the impact: probably not a fall
        self._quiet_until = self._i + self.refractory
        # Higher peak and stiller aftermath -> more confident. Range ~[0.5, 1].
        conf = 0.5 + 0.25 * min(1.0, (peak - self.impact_g) / 3.0) \
            + 0.25 * (1.0 - std / self.still_std_g)
        return Detection(self._i, conf, peak)


DETECTORS = {"threshold": ThresholdDetector}


def make_detector(name: str, fs: float) -> Detector:
    try:
        return DETECTORS[name](fs)
    except KeyError:
        raise ValueError(f"unknown detector {name!r}; choose from {sorted(DETECTORS)}")
