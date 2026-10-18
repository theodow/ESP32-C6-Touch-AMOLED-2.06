"""Virtual watches and the monitor talking through an in-process amqtt broker."""

import argparse
import asyncio
import socket
import threading
import time

import pytest

amqtt_broker = pytest.importorskip("amqtt.broker")

from fallsim import protocol as P  # noqa: E402
from fallsim.monitor import Monitor  # noqa: E402
from fallsim.simulate import parser, run  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def broker_port():
    port = _free_port()
    ready = threading.Event()
    loop = asyncio.new_event_loop()

    async def serve():
        b = amqtt_broker.Broker({
            "listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{port}"}},
            "sys_interval": 0,
            "auth": {"allow-anonymous": True},
        })
        await b.start()
        ready.set()
        await asyncio.Event().wait()

    t = threading.Thread(target=loop.run_until_complete, args=(serve(),), daemon=True)
    t.start()
    assert ready.wait(10)
    return port


def test_simulation_reaches_monitor(broker_port):
    mon = Monitor(port=broker_port, auto_ack=True, quiet=True)
    assert mon.start()
    args: argparse.Namespace = parser().parse_args([
        "--port", str(broker_port), "--devices", "4", "--speed", "50",
        "--duration", "240", "--ramp", "0", "--imu-stream", "--fall-ratio", "0.5",
        "--seed", "7",
    ])
    summary = asyncio.run(run(args))
    time.sleep(1.0)
    got = mon.summary()
    mon.stop()

    assert summary["devices"] == 4
    assert summary["falls_detected"] > 0
    assert got["devices_seen"] == 4
    assert got["devices_online"] == 0  # every watch said goodbye on close
    assert got["fall_events"] == summary["falls_detected"]
    assert got["messages"]["imu"] == summary["published"]["imu"]
    assert got["messages"][P.SIM_TRUTH] == summary["published"][P.SIM_TRUTH]
    truth = got["truth"]
    assert truth["tp"] > 0 and truth["fn"] == 0
    # the downstream path works: acks from the monitor reach the watches
    assert summary["acks_received"] > 0
