"""High-precision timestamp utilities."""

from __future__ import annotations

import time


def monotonic_ns() -> int:
    """Return monotonic clock in nanoseconds."""
    return time.monotonic_ns()


def monotonic_sec() -> float:
    """Return monotonic clock in seconds (float64 precision)."""
    return time.monotonic()


class EpisodeClock:
    """Tracks elapsed time within a single episode."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._running = False

    def start(self) -> None:
        self._start = monotonic_sec()
        self._running = True

    def elapsed(self) -> float:
        """Seconds since start()."""
        if not self._running:
            return 0.0
        return monotonic_sec() - self._start

    def stop(self) -> float:
        """Stop and return final elapsed time."""
        e = self.elapsed()
        self._running = False
        return e

    @property
    def running(self) -> bool:
        return self._running
