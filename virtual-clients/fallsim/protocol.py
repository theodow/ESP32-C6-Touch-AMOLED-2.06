"""Fall Guardian MQTT protocol v1: topics, QoS and message builders.

Every message is a UTF-8 JSON object. See PROTOCOL.md for the full spec.
Real watches (firmware) and virtual watches (this simulator) speak the same
protocol; the only difference is the optional ``sim`` block and the
``sim/truth`` topic, which real devices never publish.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

PROTOCOL_VERSION = 1
ROOT = "fallguardian/v1"

# Topic suffixes under fallguardian/v1/<device_id>/
STATUS = "status"
TELEMETRY = "telemetry"
IMU = "imu"
FALL = "event/fall"
CANCEL = "event/cancel"
CMD = "cmd"
SIM_TRUTH = "sim/truth"

QOS = {
    STATUS: 1,
    TELEMETRY: 0,
    IMU: 0,
    FALL: 1,
    CANCEL: 1,
    CMD: 1,
    SIM_TRUTH: 1,
}

RETAIN = {STATUS}


def topic(device_id: str, kind: str) -> str:
    if "/" in device_id or "+" in device_id or "#" in device_id:
        raise ValueError(f"invalid device id: {device_id!r}")
    return f"{ROOT}/{device_id}/{kind}"


def parse_topic(t: str) -> Optional[tuple[str, str]]:
    """Return (device_id, kind) for a protocol topic, or None if foreign."""
    prefix = ROOT + "/"
    if not t.startswith(prefix):
        return None
    rest = t[len(prefix):]
    device_id, sep, kind = rest.partition("/")
    if not sep or not device_id or not kind:
        return None
    return device_id, kind


def now_ms() -> int:
    return int(time.time() * 1000)


def encode(msg: dict[str, Any]) -> bytes:
    return json.dumps(msg, separators=(",", ":")).encode()


def decode(payload: bytes) -> dict[str, Any]:
    return json.loads(payload.decode())


def _base(ts: int, **fields: Any) -> dict[str, Any]:
    msg = {"v": PROTOCOL_VERSION, "ts": ts}
    msg.update({k: v for k, v in fields.items() if v is not None})
    return msg


def status_msg(online: bool, ts: Optional[int] = None, *, fw: Optional[str] = None,
               model: Optional[str] = None, fs: Optional[int] = None,
               virtual: Optional[bool] = None) -> dict[str, Any]:
    return _base(ts if ts is not None else now_ms(), online=online, fw=fw,
                 model=model, fs=fs, virtual=virtual)


def telemetry_msg(ts: int, *, battery_pct: float, rssi_dbm: int, uptime_s: int,
                  acc_mag_mean: float, acc_mag_max: float, acc_mag_std: float,
                  activity: Optional[str] = None,
                  sim: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _base(ts, battery_pct=round(battery_pct, 1), rssi_dbm=rssi_dbm,
                 uptime_s=uptime_s, acc_mag_mean=round(acc_mag_mean, 3),
                 acc_mag_max=round(acc_mag_max, 3),
                 acc_mag_std=round(acc_mag_std, 3), activity=activity, sim=sim)


def imu_msg(ts: int, *, seq: int, fs: int, acc: list[list[float]],
            gyr: Optional[list[list[float]]] = None) -> dict[str, Any]:
    """A batch of consecutive IMU samples; ``ts`` is the first sample's time."""
    return _base(ts, seq=seq, fs=fs,
                 acc=[[round(x, 4) for x in s] for s in acc],
                 gyr=None if gyr is None else [[round(x, 2) for x in s] for s in gyr])


def new_event_id() -> str:
    return uuid.uuid4().hex[:12]


def fall_msg(ts: int, *, event_id: str, confidence: float, detector: str,
             peak_g: float, sim: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _base(ts, event_id=event_id, confidence=round(confidence, 3),
                 detector=detector, peak_g=round(peak_g, 3), sim=sim)


def cancel_msg(ts: int, *, event_id: str, reason: str = "user_ok") -> dict[str, Any]:
    return _base(ts, event_id=event_id, reason=reason)


def cmd_ack(event_id: str) -> dict[str, Any]:
    return _base(now_ms(), cmd="ack", event_id=event_id)


def cmd_config(**settings: Any) -> dict[str, Any]:
    return _base(now_ms(), cmd="config", **settings)


def cmd_ping() -> dict[str, Any]:
    return _base(now_ms(), cmd="ping")


def truth_msg(ts: int, *, recording: str, activity: str, is_fall: bool,
              detected: bool, start_ts: int, end_ts: int,
              event_ids: list[str]) -> dict[str, Any]:
    return _base(ts, recording=recording, activity=activity, is_fall=is_fall,
                 detected=detected, start_ts=start_ts, end_ts=end_ts,
                 event_ids=event_ids)
