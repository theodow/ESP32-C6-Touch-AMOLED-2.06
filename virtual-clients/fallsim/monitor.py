"""Server-side monitor: watches every device, logs alerts, scores detections.

    python -m fallsim.monitor --auto-ack
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import paho.mqtt.client as mqtt

from . import protocol as P

log = logging.getLogger("fallsim.monitor")


@dataclass
class MonitorState:
    online: dict[str, bool] = field(default_factory=dict)
    messages: dict[str, int] = field(default_factory=dict)
    falls: dict[str, dict[str, Any]] = field(default_factory=dict)  # event_id -> fall
    cancelled: set[str] = field(default_factory=set)
    acked: set[str] = field(default_factory=set)
    # confusion matrix over replayed recordings (from sim/truth)
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    per_activity: dict[str, list[int]] = field(default_factory=dict)  # [n, detected]
    bad_messages: int = 0

    def summary(self) -> dict[str, Any]:
        def ratio(a: int, b: int) -> Optional[float]:
            return round(a / b, 4) if b else None
        return {
            "devices_seen": len(self.online),
            "devices_online": sum(self.online.values()),
            "messages": dict(sorted(self.messages.items())),
            "fall_events": len(self.falls),
            "cancelled": len(self.cancelled),
            "acked": len(self.acked),
            "bad_messages": self.bad_messages,
            "truth": {
                "tp": self.tp, "fn": self.fn, "fp": self.fp, "tn": self.tn,
                "sensitivity": ratio(self.tp, self.tp + self.fn),
                "specificity": ratio(self.tn, self.tn + self.fp),
                "per_activity": {k: {"n": v[0], "detected": v[1]}
                                 for k, v in sorted(self.per_activity.items())},
            },
        }


class Monitor:
    def __init__(self, host: str = "localhost", port: int = 1883,
                 username: Optional[str] = None, password: Optional[str] = None,
                 tls: bool = False, auto_ack: bool = False, client_id: str = "fallsim-monitor",
                 quiet: bool = False):
        self.state = MonitorState()
        self.auto_ack = auto_ack
        self.quiet = quiet
        self._lock = threading.Lock()
        self._subscribed = threading.Event()
        self.host, self.port = host, port
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        if username:
            self.client.username_pw_set(username, password)
        if tls:
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        self.client.on_connect = self._on_connect
        self.client.on_subscribe = lambda *a: self._subscribed.set()
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("connect refused: %s", reason_code)
            return
        client.subscribe(f"{P.ROOT}/+/#", qos=1)

    def _say(self, fmt: str, *args: Any) -> None:
        if not self.quiet:
            log.info(fmt, *args)

    def _on_message(self, client, userdata, msg):
        parsed = P.parse_topic(msg.topic)
        if parsed is None:
            return
        device, kind = parsed
        if kind == P.CMD:
            return  # our own (or another server's) downstream commands
        try:
            body = P.decode(msg.payload)
        except ValueError:
            with self._lock:
                self.state.bad_messages += 1
            return
        with self._lock:
            s = self.state
            s.messages[kind] = s.messages.get(kind, 0) + 1
            if kind == P.STATUS:
                online = bool(body.get("online"))
                if s.online.get(device) != online:
                    self._say("%s %s", device, "online" if online else "OFFLINE")
                s.online[device] = online
            elif kind == P.FALL:
                eid = body.get("event_id", "?")
                s.falls[eid] = {"device": device, **body}
                self._say("FALL %s event=%s conf=%.2f peak=%.1fg truth=%s", device, eid,
                          body.get("confidence", 0), body.get("peak_g", 0),
                          (body.get("sim") or {}).get("activity", "-"))
                if self.auto_ack:
                    client.publish(P.topic(device, P.CMD), P.encode(P.cmd_ack(eid)),
                                   qos=P.QOS[P.CMD])
                    s.acked.add(eid)
            elif kind == P.CANCEL:
                s.cancelled.add(body.get("event_id", "?"))
                self._say("cancel %s event=%s (%s)", device, body.get("event_id"),
                          body.get("reason"))
            elif kind == P.SIM_TRUTH:
                fall, det = bool(body.get("is_fall")), bool(body.get("detected"))
                if fall and det:
                    s.tp += 1
                elif fall:
                    s.fn += 1
                elif det:
                    s.fp += 1
                else:
                    s.tn += 1
                pa = s.per_activity.setdefault(str(body.get("activity")), [0, 0])
                pa[0] += 1
                pa[1] += int(det)

    def start(self, timeout: float = 10.0) -> bool:
        self.client.connect_async(self.host, self.port, 30)
        self.client.loop_start()
        return self._subscribed.wait(timeout)

    def stop(self) -> None:
        self.client.disconnect()
        self.client.loop_stop()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return self.state.summary()


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--username")
    p.add_argument("--password")
    p.add_argument("--tls", action="store_true")
    p.add_argument("--auto-ack", action="store_true", help="ack every fall event")
    p.add_argument("--duration", type=float, help="seconds to run (default: until Ctrl-C)")
    p.add_argument("--summary-every", type=float, default=0, help="print stats every N s")
    p.add_argument("--summary-json", help="write the final summary to this file")
    p.add_argument("-q", "--quiet", action="store_true", help="no per-event lines")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    m = Monitor(args.host, args.port, args.username, args.password, args.tls,
                auto_ack=args.auto_ack, quiet=args.quiet)
    if not m.start():
        raise SystemExit(f"could not subscribe on {args.host}:{args.port}")
    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *a: done.set())
    signal.signal(signal.SIGTERM, lambda *a: done.set())
    t_end = None if args.duration is None else time.monotonic() + args.duration
    next_print = time.monotonic() + args.summary_every
    while not done.is_set() and (t_end is None or time.monotonic() < t_end):
        done.wait(0.5)
        if args.summary_every and time.monotonic() >= next_print:
            print(json.dumps(m.summary()), flush=True)
            next_print += args.summary_every
    m.stop()
    summary = m.summary()
    print(json.dumps(summary, indent=2))
    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
