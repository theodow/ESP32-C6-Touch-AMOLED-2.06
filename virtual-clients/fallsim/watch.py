"""A virtual Fall Guardian watch: replays IMU recordings and speaks MQTT v1."""

from __future__ import annotations

import asyncio
import logging
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np
import paho.mqtt.client as mqtt

from . import protocol as P
from .datasets import RecordingRef, resample, still_segment
from .detector import make_detector

log = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    host: str = "localhost"
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    tls: bool = False
    keepalive: int = 30


@dataclass
class WatchConfig:
    fs: int = 50                       # IMU rate the watch runs at (Hz)
    batch_s: float = 0.5               # samples are processed/published in batches
    telemetry_interval_s: float = 5.0
    imu_stream: bool = False           # publish raw IMU batches on .../imu
    speed: float = 1.0                 # >1 replays faster than real time
    detector: str = "threshold"
    gap_s: tuple[float, float] = (3.0, 8.0)   # still time between recordings
    truth_grace_s: float = 3.0         # late detections still count for a recording
    cancel_prob_false: float = 0.7     # user presses "I'm OK" after a false alarm
    cancel_prob_true: float = 0.05     # ... or after a real fall
    cancel_delay_s: tuple[float, float] = (3.0, 15.0)
    publish_truth: bool = True         # simulation-only ground truth topic
    fw: str = "fallsim-0.1"


@dataclass
class WatchStats:
    published: dict[str, int] = field(default_factory=dict)
    recordings: int = 0
    falls_detected: int = 0
    cancels: int = 0
    acks_received: int = 0

    def count(self, kind: str) -> None:
        self.published[kind] = self.published.get(kind, 0) + 1


@dataclass
class _PendingTruth:
    ref: RecordingRef
    start_ts: int
    end_ts: int = 0
    deadline_ts: int = 0
    event_ids: list[str] = field(default_factory=list)


class VirtualWatch:
    def __init__(self, device_id: str, broker: BrokerConfig, cfg: WatchConfig,
                 playlist: list[RecordingRef], seed: int = 0):
        if not playlist:
            raise ValueError("playlist is empty")
        self.device_id = device_id
        self.broker = broker
        self.cfg = cfg
        self.playlist = playlist
        self.rng = np.random.default_rng(seed)
        self.detector = make_detector(cfg.detector, cfg.fs)
        self.stats = WatchStats()
        self.acked: set[str] = set()

        self._connected = threading.Event()
        self._t0_ms = P.now_ms()
        self._t_ms = float(self._t0_ms)  # simulated device clock
        self._battery = float(self.rng.uniform(40, 100))
        self._rssi = int(self.rng.integers(-80, -45))
        self._imu_seq = 0
        self._mags: list[float] = []
        self._next_telemetry_ms = self._t_ms + cfg.telemetry_interval_s * 1000
        self._pending_cancels: list[tuple[float, str]] = []
        self._current: Optional[_PendingTruth] = None
        self._closing: Optional[_PendingTruth] = None
        self._activity = "idle"

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=device_id)
        if broker.username:
            self.client.username_pw_set(broker.username, broker.password)
        if broker.tls:
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        self.client.will_set(P.topic(device_id, P.STATUS),
                             P.encode(P.status_msg(False, virtual=True)),
                             qos=P.QOS[P.STATUS], retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    # ---------------------------------------------------------------- MQTT

    def _status(self, online: bool) -> dict[str, Any]:
        return P.status_msg(online, int(self._t_ms), fw=self.cfg.fw,
                            model=self.detector.name, fs=self.cfg.fs, virtual=True)

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.warning("%s: connect refused: %s", self.device_id, reason_code)
            return
        client.subscribe(P.topic(self.device_id, P.CMD), qos=P.QOS[P.CMD])
        self._publish(P.STATUS, self._status(True))
        self._connected.set()

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self._connected.clear()

    def _on_message(self, client, userdata, msg):
        if msg.topic != P.topic(self.device_id, P.CMD):
            return  # some brokers (amqtt) leak other retained topics on subscribe
        try:
            body = P.decode(msg.payload)
        except ValueError:
            log.warning("%s: bad command payload", self.device_id)
            return
        cmd = body.get("cmd")
        if cmd == "ack":
            self.acked.add(str(body.get("event_id")))
            self.stats.acks_received += 1
        elif cmd == "config":
            if "telemetry_interval_s" in body:
                self.cfg.telemetry_interval_s = max(0.5, float(body["telemetry_interval_s"]))
            if "imu_stream" in body:
                self.cfg.imu_stream = bool(body["imu_stream"])
        elif cmd == "ping":
            self._publish(P.STATUS, self._status(True))
        else:
            log.info("%s: unknown command %r on %s", self.device_id, cmd, msg.topic)

    def _publish(self, kind: str, msg: dict[str, Any]) -> None:
        self.client.publish(P.topic(self.device_id, kind), P.encode(msg),
                            qos=P.QOS[kind], retain=kind in P.RETAIN)
        self.stats.count(kind)

    def connect(self, timeout: float = 10.0) -> bool:
        self.client.connect_async(self.broker.host, self.broker.port, self.broker.keepalive)
        self.client.loop_start()
        return self._connected.wait(timeout)

    def close(self) -> None:
        if self._connected.is_set():
            info = self.client.publish(P.topic(self.device_id, P.STATUS),
                                       P.encode(self._status(False)),
                                       qos=P.QOS[P.STATUS], retain=True)
            try:
                info.wait_for_publish(timeout=2.0)
            except (RuntimeError, ValueError):
                pass
        self.client.disconnect()
        self.client.loop_stop()

    # ------------------------------------------------------------ replay

    def _segments(self) -> Iterator[tuple[np.ndarray, Optional[np.ndarray], Optional[RecordingRef]]]:
        """Endless stream of (acc, gyr, ref): still gaps and recordings."""
        last = None
        while True:
            order = self.rng.permutation(len(self.playlist))
            for idx in order:
                ref = self.playlist[idx]
                gap = still_segment(self.cfg.fs, self.rng.uniform(*self.cfg.gap_s),
                                    self.rng, gravity=last)
                yield gap, None, None
                try:
                    rec = resample(ref.load(), self.cfg.fs)
                except (OSError, ValueError) as e:
                    log.warning("%s: skipping %s: %s", self.device_id, ref.name, e)
                    continue
                last = rec.acc[-1]
                yield rec.acc, rec.gyr, ref

    def _sim_block(self) -> Optional[dict[str, Any]]:
        if self._current is None:
            return {"activity": "idle", "is_fall": False}
        ref = self._current.ref
        return {"recording": ref.name, "activity": ref.activity, "is_fall": ref.is_fall,
                "source": ref.source}

    def _begin_recording(self, ref: RecordingRef) -> None:
        self._current = _PendingTruth(ref, int(self._t_ms))
        self._activity = ref.activity
        self.stats.recordings += 1

    def _end_recording(self) -> None:
        cur = self._current
        if cur is None:
            return
        cur.end_ts = int(self._t_ms)
        cur.deadline_ts = cur.end_ts + int(self.cfg.truth_grace_s * 1000)
        self._flush_truth()  # a previous one still in grace is closed now
        self._closing = cur
        self._current = None
        self._activity = "idle"

    def _flush_truth(self) -> None:
        c = self._closing
        if c is None:
            return
        self._closing = None
        if self.cfg.publish_truth:
            self._publish(P.SIM_TRUTH, P.truth_msg(
                int(self._t_ms), recording=c.ref.name, activity=c.ref.activity,
                is_fall=c.ref.is_fall, detected=bool(c.event_ids),
                start_ts=c.start_ts, end_ts=c.end_ts, event_ids=c.event_ids))

    def _on_detection(self, ts: int, confidence: float, peak_g: float) -> None:
        event_id = P.new_event_id()
        target = self._current or self._closing
        is_fall = bool(target and target.ref.is_fall)
        if target is not None:
            target.event_ids.append(event_id)
        sim = None
        if target is not None:
            sim = {"recording": target.ref.name, "activity": target.ref.activity,
                   "is_fall": is_fall}
        self._publish(P.FALL, P.fall_msg(ts, event_id=event_id, confidence=confidence,
                                         detector=self.detector.name, peak_g=peak_g,
                                         sim=sim))
        self.stats.falls_detected += 1
        p = self.cfg.cancel_prob_true if is_fall else self.cfg.cancel_prob_false
        if self.rng.random() < p:
            due = ts + self.rng.uniform(*self.cfg.cancel_delay_s) * 1000
            self._pending_cancels.append((due, event_id))

    def _process_batch(self, acc: np.ndarray, gyr: Optional[np.ndarray]) -> None:
        dt_ms = 1000.0 / self.cfg.fs
        batch_ts = int(self._t_ms)
        for k in range(len(acc)):
            det = self.detector.update(acc[k], None if gyr is None else gyr[k])
            self._mags.append(float(np.linalg.norm(acc[k])))
            if det is not None:
                self._on_detection(int(self._t_ms), det.confidence, det.peak_g)
            self._t_ms += dt_ms

        if self.cfg.imu_stream:
            self._publish(P.IMU, P.imu_msg(
                batch_ts, seq=self._imu_seq, fs=self.cfg.fs, acc=acc.tolist(),
                gyr=None if gyr is None else gyr.tolist()))
            self._imu_seq += 1

        if self._closing is not None and self._t_ms >= self._closing.deadline_ts:
            self._flush_truth()

        due = [c for c in self._pending_cancels if c[0] <= self._t_ms]
        if due:
            self._pending_cancels = [c for c in self._pending_cancels if c[0] > self._t_ms]
            for ts, event_id in due:
                self._publish(P.CANCEL, P.cancel_msg(int(ts), event_id=event_id))
                self.stats.cancels += 1

        if self._t_ms >= self._next_telemetry_ms:
            self._publish_telemetry()

    def _publish_telemetry(self) -> None:
        mags = np.asarray(self._mags) if self._mags else np.ones(1)
        self._mags = []
        elapsed_h = self.cfg.telemetry_interval_s / 3600
        self._battery = max(0.0, self._battery - elapsed_h * self.rng.uniform(3, 6))
        self._rssi = int(np.clip(self._rssi + self.rng.integers(-2, 3), -95, -35))
        self._publish(P.TELEMETRY, P.telemetry_msg(
            int(self._t_ms), battery_pct=self._battery, rssi_dbm=self._rssi,
            uptime_s=int((self._t_ms - self._t0_ms) / 1000),
            acc_mag_mean=float(mags.mean()), acc_mag_max=float(mags.max()),
            acc_mag_std=float(mags.std()), activity=self._activity,
            sim=self._sim_block()))
        self._next_telemetry_ms = self._t_ms + self.cfg.telemetry_interval_s * 1000

    async def run(self, duration_s: Optional[float] = None,
                  stop: Optional[asyncio.Event] = None) -> WatchStats:
        """Replay until ``duration_s`` of simulated time has passed or ``stop``."""
        batch_n = max(1, int(self.cfg.batch_s * self.cfg.fs))
        end_ms = None if duration_s is None else self._t_ms + duration_s * 1000
        wall0 = time.monotonic()
        sim0 = self._t_ms
        for acc, gyr, ref in self._segments():
            if ref is not None:
                self._begin_recording(ref)
            for i in range(0, len(acc), batch_n):
                if (stop is not None and stop.is_set()) or \
                        (end_ms is not None and self._t_ms >= end_ms):
                    return self.stats
                self._process_batch(acc[i:i + batch_n],
                                    None if gyr is None else gyr[i:i + batch_n])
                # pace against the wall clock so lag does not accumulate
                target = wall0 + (self._t_ms - sim0) / 1000 / self.cfg.speed
                delay = target - time.monotonic()
                await asyncio.sleep(max(0.0, delay))
            if ref is not None:
                self._end_recording()
        return self.stats
