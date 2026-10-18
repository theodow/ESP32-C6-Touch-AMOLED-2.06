"""Tests that need no broker: protocol, dataset parsers, detector."""

from pathlib import Path

import numpy as np
import pytest

from fallsim import protocol as P
from fallsim.datasets import (list_mobifall, list_sisfall, list_synthetic,
                              parse_sisfall_file, resample, still_segment)
from fallsim.detector import ThresholdDetector


def test_topics_roundtrip():
    t = P.topic("vw-0001", P.FALL)
    assert t == "fallguardian/v1/vw-0001/event/fall"
    assert P.parse_topic(t) == ("vw-0001", "event/fall")
    assert P.parse_topic("other/v1/x/status") is None
    with pytest.raises(ValueError):
        P.topic("bad/id", P.STATUS)


def test_messages_are_versioned_json():
    msg = P.fall_msg(123, event_id="abc", confidence=0.91234, detector="d", peak_g=3.21)
    back = P.decode(P.encode(msg))
    assert back["v"] == P.PROTOCOL_VERSION and back["ts"] == 123
    assert back["confidence"] == 0.912
    assert "sim" not in back  # optional fields are omitted, not null


def test_sisfall_parser(tmp_path: Path):
    subj = tmp_path / "SA01"
    subj.mkdir()
    # 1 g on the ADXL345 y axis = 256 counts; 2000/2^15 deg/s per count
    rows = ["0, 256, 0, 16, 0, 0, 0, 1024, 0;"] * 400
    (subj / "F05_SA01_R01.txt").write_text("\n".join(rows) + "\n")
    (subj / "D01_SA01_R01.txt").write_text("\n".join(rows) + "\n")
    (subj / "readme.txt").write_text("not a recording")
    refs = list_sisfall(tmp_path)
    assert [(r.activity, r.is_fall, r.subject) for r in refs] == \
        [("D01", False, "SA01"), ("F05", True, "SA01")]
    rec = refs[1].load()
    assert rec.fs == 200 and rec.acc.shape == (400, 3)
    assert rec.acc[0, 1] == pytest.approx(1.0)
    assert rec.gyr[0, 0] == pytest.approx(16 * 4000 / 2 ** 16)
    mma = parse_sisfall_file(subj / "F05_SA01_R01.txt", accel="mma8451q")
    assert mma.acc[0, 1] == pytest.approx(1.0)


def _mobifall_file(path: Path, rows: list[str]) -> None:
    path.write_text("#Activity: Forward-lying\n#Subject ID: 1\n\n@DATA\n" + "\n".join(rows) + "\n")


def test_mobifall_parser(tmp_path: Path):
    d = tmp_path / "sub1" / "FALLS" / "FOL"
    d.mkdir(parents=True)
    # ~100 Hz with jitter, 2 s, gravity on z in m/s^2
    ts = np.cumsum(np.full(200, 10_000_000) + np.arange(200) % 3 * 1_000_000)
    _mobifall_file(d / "FOL_acc_1_1.txt", [f"{t},0.0,0.0,9.80665" for t in ts])
    _mobifall_file(d / "FOL_gyro_1_1.txt", [f"{t},{np.pi},0.0,0.0" for t in ts])
    refs = list_mobifall(tmp_path, fs=50)
    assert len(refs) == 1 and refs[0].is_fall and refs[0].subject == "sub1"
    rec = refs[0].load()
    assert rec.fs == 50
    assert np.allclose(rec.acc[:, 2], 1.0)
    assert np.allclose(rec.gyr[:, 0], 180.0)


def test_resample_keeps_duration():
    rec = list_synthetic(1, seed=1)[0].load()
    out = resample(rec, 25)
    assert out.fs == 25
    assert out.duration_s == pytest.approx(rec.duration_s, abs=0.1)


def test_still_segment_follows_gravity():
    seg = still_segment(50, 1.0, np.random.default_rng(0), gravity=np.array([0, 2.0, 0]))
    assert np.allclose(seg.mean(axis=0), [0, 1, 0], atol=0.01)


def _detects(rec) -> bool:
    d = ThresholdDetector(rec.fs)
    tail = np.tile(rec.acc[-1], (int(3 * rec.fs), 1))
    return any(d.update(s) is not None for s in np.vstack([rec.acc, tail]))


def test_threshold_detector_on_synthetic():
    refs = list_synthetic(120, fall_ratio=0.4, seed=3)
    falls = [_detects(r.load()) for r in refs if r.is_fall]
    calm = [_detects(r.load()) for r in refs if r.activity in ("STD", "WAL", "SIT", "JOG")]
    assert sum(falls) / len(falls) > 0.95
    assert sum(calm) == 0
