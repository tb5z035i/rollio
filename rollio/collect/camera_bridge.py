"""Thread-backed adapters for blocking camera backends."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from rollio.sensors import ImageSensor


@dataclass(frozen=True)
class FrameSample:
    """One captured frame with its capture timestamp."""

    timestamp: float
    frame: np.ndarray


@dataclass(frozen=True)
class FrameSourceMetrics:
    """Basic capture metrics for one threaded frame source."""

    captured_frames: int = 0
    dropped_frames: int = 0
    last_capture_timestamp: float | None = None


class ThreadedCameraFrameSource:
    """Continuously captures frames in a background thread.

    The scheduler can poll this source cheaply without blocking on camera I/O.
    """

    def __init__(
        self,
        name: str,
        camera: ImageSensor,
        *,
        max_pending_frames: int = 4,
    ) -> None:
        self.name = name
        self.camera = camera
        self.fps = max(int(camera.fps), 1)
        self._interval_sec = 1.0 / self.fps
        self._max_pending_frames = max(1, max_pending_frames)
        self._lock = threading.Lock()
        self._pending: deque[FrameSample] = deque(maxlen=self._max_pending_frames)
        self._latest: FrameSample | None = None
        self._captured_frames = 0
        self._dropped_frames = 0
        self._last_capture_timestamp: float | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._open_error: Exception | None = None
        self._is_open = False

    def open(self) -> None:
        if self._is_open:
            return
        self._stop_event.clear()
        self._ready_event.clear()
        self._open_error = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-source-{self.name}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=5.0):
            raise RuntimeError(f"Timed out while opening camera source '{self.name}'")
        if self._open_error is not None:
            raise RuntimeError(
                f"Failed to open camera source '{self.name}': {self._open_error}"
            ) from self._open_error
        self._is_open = True

    def close(self) -> None:
        if not self._is_open:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                # Best-effort fallback for backends whose read loop does not wake
                # promptly on its own.
                self.camera.close()
                self._thread.join(timeout=5.0)
        self._thread = None
        self._is_open = False

    def latest_sample(self) -> FrameSample | None:
        with self._lock:
            return self._latest

    def drain_samples(self) -> list[FrameSample]:
        with self._lock:
            samples = list(self._pending)
            self._pending.clear()
        return samples

    def metrics(self) -> FrameSourceMetrics:
        with self._lock:
            return FrameSourceMetrics(
                captured_frames=self._captured_frames,
                dropped_frames=self._dropped_frames,
                last_capture_timestamp=self._last_capture_timestamp,
            )

    def _run(self) -> None:
        try:
            self.camera.open()
        except Exception as exc:
            self._open_error = exc
            self._ready_event.set()
            return

        self._ready_event.set()
        next_tick = time.monotonic()
        try:
            while not self._stop_event.is_set():
                ts, frame = self.camera.read()
                sample = FrameSample(timestamp=ts, frame=frame)
                with self._lock:
                    if len(self._pending) == self._max_pending_frames:
                        self._dropped_frames += 1
                    self._pending.append(sample)
                    self._latest = sample
                    self._captured_frames += 1
                    self._last_capture_timestamp = ts

                next_tick += self._interval_sec
                remaining = next_tick - time.monotonic()
                if remaining <= 0:
                    next_tick = time.monotonic()
                    continue
                if self._stop_event.wait(remaining):
                    return
        finally:
            try:
                self.camera.close()
            except Exception:
                pass
