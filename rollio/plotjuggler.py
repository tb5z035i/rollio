"""Best-effort PlotJuggler UDP streaming helpers."""

from __future__ import annotations

import atexit
import json
import os
import queue
import socket
import threading
from dataclasses import dataclass

DEFAULT_PLOTJUGGLER_PORT = 9870
PLOTJUGGLER_HOST = "127.0.0.1"
PLOTJUGGLER_PORT_ENV = "PLOTJUGGLER_PORT"


def _get_plotjuggler_port() -> int:
    raw = str(os.getenv(PLOTJUGGLER_PORT_ENV, str(DEFAULT_PLOTJUGGLER_PORT))).strip()
    try:
        port = int(raw)
    except ValueError:
        return DEFAULT_PLOTJUGGLER_PORT
    if 1 <= port <= 65535:
        return port
    return DEFAULT_PLOTJUGGLER_PORT


def _build_plotjuggler_message(
    robot_name: str,
    timestamp: float,
    position: tuple[float, ...],
) -> dict[str, object]:
    return {
        "timestamp": float(timestamp),
        str(robot_name): {
            f"j{idx}": float(value) for idx, value in enumerate(position)
        },
    }


def _encode_plotjuggler_message(
    robot_name: str,
    timestamp: float,
    position: tuple[float, ...],
) -> bytes:
    return json.dumps(
        _build_plotjuggler_message(robot_name, timestamp, position),
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class _JointSample:
    robot_name: str
    timestamp: float
    position: tuple[float, ...]


class _PlotJugglerUdpPublisher:
    """Background UDP publisher so JSON/socket work stays off read paths."""

    def __init__(self, host: str, port: int) -> None:
        self._address = (str(host), int(port))
        self._queue: queue.SimpleQueue[_JointSample | None] = queue.SimpleQueue()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._thread = threading.Thread(
            target=self._run,
            name="plotjuggler-udp-publisher",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    def publish_joint_state(
        self,
        robot_name: str,
        timestamp: float,
        position: tuple[float, ...],
    ) -> None:
        if self._closed:
            return
        self._queue.put(
            _JointSample(
                robot_name=str(robot_name),
                timestamp=float(timestamp),
                position=tuple(float(value) for value in position),
            )
        )

    def _run(self) -> None:
        while True:
            sample = self._queue.get()
            if sample is None:
                break
            try:
                self._sock.sendto(
                    _encode_plotjuggler_message(
                        sample.robot_name,
                        sample.timestamp,
                        sample.position,
                    ),
                    self._address,
                )
            except (BlockingIOError, InterruptedError, OSError):
                continue

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=1.0)
        self._sock.close()


_PLOTJUGGLER_STATE: dict[str, _PlotJugglerUdpPublisher | None] = {"publisher": None}
_PLOTJUGGLER_PUBLISHER_LOCK = threading.Lock()


def _get_plotjuggler_publisher() -> _PlotJugglerUdpPublisher:
    with _PLOTJUGGLER_PUBLISHER_LOCK:
        publisher = _PLOTJUGGLER_STATE["publisher"]
        if publisher is None:
            publisher = _PlotJugglerUdpPublisher(
                PLOTJUGGLER_HOST,
                _get_plotjuggler_port(),
            )
            _PLOTJUGGLER_STATE["publisher"] = publisher
        return publisher


def publish_joint_state(
    robot_name: str,
    timestamp: float,
    position: tuple[float, ...],
) -> None:
    """Queue one robot joint-state sample for best-effort UDP publishing."""
    if not robot_name or not position:
        return
    _get_plotjuggler_publisher().publish_joint_state(robot_name, timestamp, position)


def close_plotjuggler_publisher() -> None:
    """Close the shared PlotJuggler UDP publisher if it was created."""
    with _PLOTJUGGLER_PUBLISHER_LOCK:
        publisher = _PLOTJUGGLER_STATE["publisher"]
        _PLOTJUGGLER_STATE["publisher"] = None
    if publisher is not None:
        publisher.close()


atexit.register(close_plotjuggler_publisher)

