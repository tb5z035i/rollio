"""Internal helpers for AIRBOT command replay."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

import numpy as np

from rollio.robot.base import ControlMode, Wrench


def clone_wrench(wrench: Wrench | None) -> Wrench | None:
    """Return a detached copy of one wrench payload."""
    if wrench is None:
        return None
    return Wrench(
        force=np.asarray(wrench.force, dtype=np.float64).copy(),
        torque=np.asarray(wrench.torque, dtype=np.float64).copy(),
    )


@dataclass(frozen=True)
class AirbotFreeDriveIntent:
    """Latest free-drive intent published by the caller."""

    gravity_compensation_scale: float = 1.0
    external_wrench: Wrench | None = None
    published_at: float = 0.0


@dataclass(frozen=True)
class AirbotFixedTrackingIntent:
    """Latest fixed tracking packet that should be replayed at the loop rate."""

    position_target: np.ndarray
    velocity_target: np.ndarray
    feedforward: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    published_at: float = 0.0


@dataclass(frozen=True)
class AirbotDynamicTrackingIntent:
    """Latest high-level target that should refresh feedforward each tick."""

    position_target: np.ndarray
    velocity_target: np.ndarray
    user_feedforward: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    add_gravity_compensation: bool = True
    published_at: float = 0.0


@dataclass(frozen=True)
class AirbotPvtCommand:
    """Raw PVT command replayed directly by the command pump."""

    position_target: np.ndarray
    velocity_target: np.ndarray
    effort: np.ndarray
    published_at: float = 0.0


AirbotCommand = (
    AirbotFreeDriveIntent
    | AirbotFixedTrackingIntent
    | AirbotDynamicTrackingIntent
    | AirbotPvtCommand
)


@dataclass(frozen=True)
class AirbotLoopMetrics:
    """Fixed-rate loop timing snapshot for one AIRBOT device thread."""

    target_interval_ms: float
    run_count: int
    last_interval_ms: float | None = None
    avg_interval_ms: float | None = None
    max_interval_ms: float | None = None


@dataclass(frozen=True)
class _EnableRequest:
    seq: int
    enabled: bool


@dataclass(frozen=True)
class _ModeRequest:
    seq: int
    mode: ControlMode


class AirbotCommandLease:
    """Exclusive ownership token for temporary command-slot takeover."""

    def __init__(self, pump: "AirbotCommandPump", owner: str) -> None:
        self._pump = pump
        self._owner = owner
        self._released = False

    @property
    def owner(self) -> str:
        return self._owner

    def close(self) -> None:
        if self._released:
            return
        self._pump.release_lease(self._owner)
        self._released = True

    def __enter__(self) -> "AirbotCommandLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        self.close()
        return False


class AirbotCommandPump:
    """One fixed-rate thread that owns AIRBOT write-side I/O."""

    def __init__(
        self,
        *,
        name: str,
        period_sec: float,
        apply_enabled: Callable[[bool], bool],
        apply_mode: Callable[[ControlMode], bool],
        cycle: Callable[[AirbotCommand | None, ControlMode, bool], None],
        initial_enabled: bool = False,
        initial_mode: ControlMode = ControlMode.DISABLED,
    ) -> None:
        self._name = name
        self._period_sec = max(1e-4, float(period_sec))
        self._apply_enabled = apply_enabled
        self._apply_mode = apply_mode
        self._cycle = cycle
        self._lock = threading.Lock()
        self._latest_command: AirbotCommand | None = None
        self._lease_owner: str | None = None
        self._applied_enabled = bool(initial_enabled)
        self._applied_mode = initial_mode if initial_enabled else ControlMode.DISABLED
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._enable_result_event = threading.Event()
        self._mode_result_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._request_seq = 0
        self._enable_request: _EnableRequest | None = None
        self._enable_result_seq = 0
        self._enable_result_ok = False
        self._mode_request: _ModeRequest | None = None
        self._mode_result_seq = 0
        self._mode_result_ok = False
        self._handled_enable_seq = 0
        self._handled_mode_seq = 0
        self._run_count = 0
        self._last_tick_at: float | None = None
        self._last_interval_ms: float | None = None
        self._avg_interval_ms: float | None = None
        self._max_interval_ms: float | None = None

    @property
    def applied_enabled(self) -> bool:
        return self._applied_enabled

    @property
    def applied_mode(self) -> ControlMode:
        return self._applied_mode

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=self._name,
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        self._wake_event.set()
        thread.join(timeout=max(timeout, self._period_sec))
        self._thread = None
        with self._lock:
            self._latest_command = None
            self._lease_owner = None

    def metrics(self) -> AirbotLoopMetrics:
        return AirbotLoopMetrics(
            target_interval_ms=self._period_sec * 1000.0,
            run_count=self._run_count,
            last_interval_ms=self._last_interval_ms,
            avg_interval_ms=self._avg_interval_ms,
            max_interval_ms=self._max_interval_ms,
        )

    def publish_command(
        self,
        command: AirbotCommand | None,
        *,
        owner: str | None = None,
    ) -> bool:
        with self._lock:
            if self._lease_owner is not None and owner != self._lease_owner:
                return False
            self._latest_command = command
        self._wake_event.set()
        return True

    def reset_command(self) -> None:
        with self._lock:
            self._latest_command = None
        self._wake_event.set()

    def snapshot_command(self) -> AirbotCommand | None:
        with self._lock:
            return self._latest_command

    def acquire_lease(self, owner: str) -> AirbotCommandLease | None:
        normalized_owner = str(owner).strip()
        if not normalized_owner:
            raise ValueError("Lease owner must be a non-empty string")
        with self._lock:
            if self._lease_owner not in (None, normalized_owner):
                return None
            self._lease_owner = normalized_owner
        self._wake_event.set()
        return AirbotCommandLease(self, normalized_owner)

    def release_lease(self, owner: str) -> None:
        normalized_owner = str(owner).strip()
        with self._lock:
            if self._lease_owner == normalized_owner:
                self._lease_owner = None
        self._wake_event.set()

    def request_enabled(self, enabled: bool, timeout: float = 2.0) -> bool:
        if self._thread is None:
            return False
        if bool(enabled) == self._applied_enabled:
            return True
        seq = self._next_request_seq()
        self._enable_result_event.clear()
        self._enable_request = _EnableRequest(seq=seq, enabled=bool(enabled))
        self._wake_event.set()
        return self._await_enable_result(seq, timeout)

    def request_mode(self, mode: ControlMode, timeout: float = 2.0) -> bool:
        if self._thread is None:
            return False
        if mode == self._applied_mode:
            return True
        if mode != ControlMode.DISABLED and not self._applied_enabled:
            return False
        seq = self._next_request_seq()
        self._mode_result_event.clear()
        self._mode_request = _ModeRequest(seq=seq, mode=mode)
        self._wake_event.set()
        return self._await_mode_result(seq, timeout)

    def _next_request_seq(self) -> int:
        self._request_seq += 1
        return self._request_seq

    def _await_enable_result(self, seq: int, timeout: float) -> bool:
        deadline = monotonic() + max(timeout, self._period_sec)
        while monotonic() < deadline:
            if self._enable_result_seq == seq:
                return self._enable_result_ok
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            self._enable_result_event.wait(min(remaining, 0.05))
            self._enable_result_event.clear()
        return False

    def _await_mode_result(self, seq: int, timeout: float) -> bool:
        deadline = monotonic() + max(timeout, self._period_sec)
        while monotonic() < deadline:
            if self._mode_result_seq == seq:
                return self._mode_result_ok
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            self._mode_result_event.wait(min(remaining, 0.05))
            self._mode_result_event.clear()
        return False

    def _run(self) -> None:
        next_tick = monotonic()
        while not self._stop_event.is_set():
            now = monotonic()
            if now < next_tick:
                self._wake_event.wait(next_tick - now)
                self._wake_event.clear()
                continue

            self._observe_interval(now)
            self._process_enable_request()
            self._process_mode_request()

            try:
                self._cycle(
                    self.snapshot_command(),
                    self._applied_mode,
                    self._applied_enabled,
                )
            except Exception:
                pass

            next_tick += self._period_sec
            if next_tick <= monotonic() - self._period_sec:
                next_tick = monotonic() + self._period_sec

    def _observe_interval(self, now: float) -> None:
        if self._last_tick_at is not None:
            interval_ms = max(0.0, (now - self._last_tick_at) * 1000.0)
            self._last_interval_ms = interval_ms
            self._max_interval_ms = (
                interval_ms
                if self._max_interval_ms is None
                else max(self._max_interval_ms, interval_ms)
            )
            if self._avg_interval_ms is None:
                self._avg_interval_ms = interval_ms
            else:
                prev = max(self._run_count - 1, 1)
                self._avg_interval_ms = (
                    (self._avg_interval_ms * prev) + interval_ms
                ) / self._run_count
        self._last_tick_at = now
        self._run_count += 1

    def _process_enable_request(self) -> None:
        request = self._enable_request
        if request is None or request.seq == self._handled_enable_seq:
            return
        ok = False
        try:
            ok = self._apply_enabled(request.enabled)
        except Exception:
            ok = False
        if ok:
            self._applied_enabled = request.enabled
            if not request.enabled:
                self._applied_mode = ControlMode.DISABLED
        self._handled_enable_seq = request.seq
        self._enable_result_seq = request.seq
        self._enable_result_ok = ok
        self._enable_result_event.set()

    def _process_mode_request(self) -> None:
        request = self._mode_request
        if request is None or request.seq == self._handled_mode_seq:
            return
        ok = False
        try:
            if request.mode == ControlMode.DISABLED:
                ok = self._apply_mode(request.mode) if self._applied_enabled else True
            elif self._applied_enabled:
                ok = self._apply_mode(request.mode)
        except Exception:
            ok = False
        if ok:
            self._applied_mode = request.mode
        self._handled_mode_seq = request.seq
        self._mode_result_seq = request.seq
        self._mode_result_ok = ok
        self._mode_result_event.set()


__all__ = [
    "AirbotCommandLease",
    "AirbotCommandPump",
    "AirbotDynamicTrackingIntent",
    "AirbotFixedTrackingIntent",
    "AirbotFreeDriveIntent",
    "AirbotLoopMetrics",
    "AirbotPvtCommand",
    "clone_wrench",
]
