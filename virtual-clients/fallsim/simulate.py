"""Run many virtual watches against an MQTT broker.

    python -m fallsim.simulate --devices 20 --dataset synthetic --speed 5
    python -m fallsim.simulate --devices 50 --dataset sisfall --data-root ~/data/SisFall_dataset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from typing import Optional

import numpy as np

from .datasets import RecordingRef, list_mobifall, list_sisfall, list_synthetic
from .detector import DETECTORS
from .watch import BrokerConfig, VirtualWatch, WatchConfig

log = logging.getLogger("fallsim")


def load_refs(args: argparse.Namespace) -> list[RecordingRef]:
    if args.dataset == "synthetic":
        refs = list_synthetic(count=args.synthetic_count, fall_ratio=args.fall_ratio,
                              fs=args.fs, seed=args.seed)
    elif args.dataset == "sisfall":
        refs = list_sisfall(_need_root(args), accel=args.sisfall_accel)
    else:
        refs = list_mobifall(_need_root(args), fs=args.fs)
    if not refs:
        sys.exit(f"no recordings found for {args.dataset} under {args.data_root}")
    if args.only == "falls":
        refs = [r for r in refs if r.is_fall]
    elif args.only == "adl":
        refs = [r for r in refs if not r.is_fall]
    if args.subjects:
        wanted = set(args.subjects.split(","))
        refs = [r for r in refs if r.subject in wanted]
    if not refs:
        sys.exit("no recordings left after filtering")
    return refs


def _need_root(args: argparse.Namespace) -> str:
    if not args.data_root:
        sys.exit(f"--data-root is required for --dataset {args.dataset}")
    return args.data_root


def build_watches(args: argparse.Namespace, refs: list[RecordingRef]) -> list[VirtualWatch]:
    broker = BrokerConfig(args.host, args.port, args.username, args.password, args.tls)
    rng = np.random.default_rng(args.seed)
    watches = []
    for i in range(args.devices):
        cfg = WatchConfig(fs=args.fs, telemetry_interval_s=args.telemetry_interval,
                          imu_stream=args.imu_stream, speed=args.speed,
                          detector=args.detector, publish_truth=not args.no_truth)
        # each watch replays its own subject-sized slice of the data, cycling
        idx = rng.permutation(len(refs))[:max(1, args.recordings_per_device)]
        watches.append(VirtualWatch(f"{args.prefix}{i:04d}", broker, cfg,
                                    [refs[j] for j in idx], seed=args.seed + i + 1))
    return watches


async def run(args: argparse.Namespace, stop: Optional[asyncio.Event] = None) -> dict:
    refs = load_refs(args)
    falls = sum(r.is_fall for r in refs)
    log.info("%d recordings (%d falls) from %s", len(refs), falls, args.dataset)
    watches = build_watches(args, refs)
    stop = stop or asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass

    connected = []
    for i, w in enumerate(watches):
        ok = await asyncio.to_thread(w.connect, 10.0)
        if not ok:
            log.error("%s could not connect to %s:%d", w.device_id, args.host, args.port)
            w.close()
            continue
        connected.append(w)
        if args.ramp > 0 and len(watches) > 1:
            await asyncio.sleep(args.ramp / len(watches))
    log.info("%d/%d watches online", len(connected), len(watches))

    try:
        await asyncio.gather(*(w.run(args.duration, stop) for w in connected))
    finally:
        for w in connected:
            w.close()

    summary = {
        "devices": len(connected),
        "recordings": sum(w.stats.recordings for w in connected),
        "falls_detected": sum(w.stats.falls_detected for w in connected),
        "cancels": sum(w.stats.cancels for w in connected),
        "acks_received": sum(w.stats.acks_received for w in connected),
        "published": {},
    }
    for w in connected:
        for k, v in w.stats.published.items():
            summary["published"][k] = summary["published"].get(k, 0) + v
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    b = p.add_argument_group("broker")
    b.add_argument("--host", default="localhost")
    b.add_argument("--port", type=int, default=1883)
    b.add_argument("--username")
    b.add_argument("--password")
    b.add_argument("--tls", action="store_true")

    d = p.add_argument_group("data")
    d.add_argument("--dataset", choices=["synthetic", "sisfall", "mobifall"], default="synthetic")
    d.add_argument("--data-root", help="dataset folder (SisFall_dataset or MobiFall_Dataset_v2.0)")
    d.add_argument("--sisfall-accel", choices=["adxl345", "mma8451q"], default="adxl345")
    d.add_argument("--only", choices=["all", "falls", "adl"], default="all")
    d.add_argument("--subjects", help="comma-separated subject ids, e.g. SA01,SE06")
    d.add_argument("--synthetic-count", type=int, default=200)
    d.add_argument("--fall-ratio", type=float, default=0.3, help="synthetic only")

    s = p.add_argument_group("simulation")
    s.add_argument("--devices", type=int, default=10)
    s.add_argument("--prefix", default="vw-", help="device id prefix")
    s.add_argument("--recordings-per-device", type=int, default=30)
    s.add_argument("--fs", type=int, default=50, help="watch IMU rate in Hz")
    s.add_argument("--duration", type=float, help="simulated seconds (default: until Ctrl-C)")
    s.add_argument("--speed", type=float, default=1.0, help="replay speed vs real time")
    s.add_argument("--ramp", type=float, default=2.0, help="seconds to spread connects over")
    s.add_argument("--telemetry-interval", type=float, default=5.0)
    s.add_argument("--imu-stream", action="store_true", help="also publish raw IMU batches")
    s.add_argument("--detector", choices=sorted(DETECTORS), default="threshold")
    s.add_argument("--no-truth", action="store_true", help="do not publish sim/truth")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--summary-json", help="write a run summary to this file")
    s.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = asyncio.run(run(args))
    print(json.dumps(summary, indent=2))
    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
