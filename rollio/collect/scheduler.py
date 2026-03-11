"""Shared scheduler drivers for collection and preview."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledTask:
    """One periodic task managed by a scheduler driver."""

    name: str
    interval_sec: float
    step: Callable[[], None]


@dataclass(frozen=True)
class TaskMetrics:
    """Observed execution statistics for one scheduled task."""

    interval_sec: float
    run_count: int
    overrun_count: int
    error_count: int
    last_step_ms: float
    avg_step_ms: float
    max_step_ms: float
    last_error: str | None = None


@dataclass(frozen=True)
class DriverMetrics:
    """Snapshot of driver-level and task-level scheduling metrics."""

    driver_name: str
    task_metrics: dict[str, TaskMetrics]
    loop_run_count: int = 0
    last_loop_us: float = 0.0
    avg_loop_us: float = 0.0
    max_loop_us: float = 0.0


@dataclass
class _MutableTaskMetrics:
    interval_sec: float
    run_count: int = 0
    overrun_count: int = 0
    error_count: int = 0
    last_step_ms: float = 0.0
    avg_step_ms: float = 0.0
    max_step_ms: float = 0.0
    last_error: str | None = None

    def observe(
        self, elapsed_sec: float, *, overrun_count: int = 0, error: str | None = None
    ) -> None:
        self.run_count += 1
        self.overrun_count += max(0, overrun_count)
        if error is not None:
            self.error_count += 1
            self.last_error = error
        elapsed_ms = elapsed_sec * 1000.0
        self.last_step_ms = elapsed_ms
        self.max_step_ms = max(self.max_step_ms, elapsed_ms)
        if self.run_count == 1:
            self.avg_step_ms = elapsed_ms
        else:
            prev = self.run_count - 1
            self.avg_step_ms = ((self.avg_step_ms * prev) + elapsed_ms) / self.run_count

    def freeze(self) -> TaskMetrics:
        return TaskMetrics(
            interval_sec=self.interval_sec,
            run_count=self.run_count,
            overrun_count=self.overrun_count,
            error_count=self.error_count,
            last_step_ms=self.last_step_ms,
            avg_step_ms=self.avg_step_ms,
            max_step_ms=self.max_step_ms,
            last_error=self.last_error,
        )


@dataclass
class _MutableLoopMetrics:
    run_count: int = 0
    last_loop_us: float = 0.0
    avg_loop_us: float = 0.0
    max_loop_us: float = 0.0

    def observe(self, elapsed_sec: float) -> None:
        elapsed_us = elapsed_sec * 1_000_000.0
        self.run_count += 1
        self.last_loop_us = elapsed_us
        self.max_loop_us = max(self.max_loop_us, elapsed_us)
        if self.run_count == 1:
            self.avg_loop_us = elapsed_us
        else:
            prev = self.run_count - 1
            self.avg_loop_us = ((self.avg_loop_us * prev) + elapsed_us) / self.run_count


@dataclass
class _TaskState:
    task: ScheduledTask
    next_run: float
    metrics: _MutableTaskMetrics


class BaseSchedulerDriver:
    """Common scheduling logic shared by both driver implementations."""

    DRIVER_NAME = "base"

    def __init__(self, tasks: list[ScheduledTask]) -> None:
        self._tasks = tasks
        self._metrics_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._loop_metrics = _MutableLoopMetrics()
        self._states: list[_TaskState] = [
            _TaskState(
                task=task,
                next_run=time.monotonic(),
                metrics=_MutableTaskMetrics(interval_sec=task.interval_sec),
            )
            for task in tasks
        ]

    def start(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def metrics(self) -> DriverMetrics:
        with self._metrics_lock:
            task_metrics = {
                state.task.name: state.metrics.freeze() for state in self._states
            }
            loop_metrics = _MutableLoopMetrics(
                run_count=self._loop_metrics.run_count,
                last_loop_us=self._loop_metrics.last_loop_us,
                avg_loop_us=self._loop_metrics.avg_loop_us,
                max_loop_us=self._loop_metrics.max_loop_us,
            )
        return DriverMetrics(
            driver_name=self.DRIVER_NAME,
            task_metrics=task_metrics,
            loop_run_count=loop_metrics.run_count,
            last_loop_us=loop_metrics.last_loop_us,
            avg_loop_us=loop_metrics.avg_loop_us,
            max_loop_us=loop_metrics.max_loop_us,
        )

    def _observe_loop(self, elapsed_sec: float) -> None:
        with self._metrics_lock:
            self._loop_metrics.observe(elapsed_sec)

    def _run_due_tasks(self) -> tuple[bool, float | None]:
        now = time.monotonic()
        ran_any = False
        next_deadline: float | None = None

        for state in self._states:
            overrun_count = 0
            if now >= state.next_run:
                if state.task.interval_sec > 0:
                    lag = max(0.0, now - state.next_run)
                    overrun_count = int(lag // state.task.interval_sec)

                started_at = time.monotonic()
                error_message: str | None = None
                try:
                    state.task.step()
                except Exception as exc:  # pragma: no cover - surfaced in metrics/tests
                    error_message = str(exc)
                elapsed = time.monotonic() - started_at
                with self._metrics_lock:
                    state.metrics.observe(
                        elapsed,
                        overrun_count=overrun_count,
                        error=error_message,
                    )
                state.next_run += (overrun_count + 1) * state.task.interval_sec
                ran_any = True

            next_deadline = (
                state.next_run
                if next_deadline is None
                else min(next_deadline, state.next_run)
            )

        return ran_any, next_deadline


class RoundRobinDriver(BaseSchedulerDriver):
    """Deterministic single-thread driver using blocking waits."""

    DRIVER_NAME = "round_robin"

    def __init__(self, tasks: list[ScheduledTask]) -> None:
        super().__init__(tasks)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="rollio-round-robin-driver",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            loop_started_at = time.monotonic()
            ran_any, next_deadline = self._run_due_tasks()
            if ran_any:
                self._observe_loop(time.monotonic() - loop_started_at)
            if ran_any:
                continue
            sleep_sec = (
                0.05
                if next_deadline is None
                else max(
                    0.0,
                    min(next_deadline - time.monotonic(), 0.05),
                )
            )
            if self._stop_event.wait(sleep_sec):
                return


class AsyncioDriver(BaseSchedulerDriver):
    """Single event-loop driver that cooperatively steps ordered tasks."""

    DRIVER_NAME = "asyncio"

    def __init__(self, tasks: list[ScheduledTask]) -> None:
        super().__init__(tasks)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name="rollio-asyncio-driver",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._loop = None

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run())
        finally:
            loop.close()
            self._loop = None

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            loop_started_at = time.monotonic()
            ran_any, next_deadline = self._run_due_tasks()
            if ran_any:
                self._observe_loop(time.monotonic() - loop_started_at)
            if ran_any:
                await asyncio.sleep(0)
                continue
            sleep_sec = (
                0.05
                if next_deadline is None
                else max(
                    0.0,
                    min(next_deadline - time.monotonic(), 0.05),
                )
            )
            await asyncio.sleep(sleep_sec)


def build_scheduler_driver(
    name: str, tasks: list[ScheduledTask]
) -> BaseSchedulerDriver:
    """Create one scheduler driver by name."""
    driver_name = str(name).strip().lower()
    if driver_name == "asyncio":
        return AsyncioDriver(tasks)
    if driver_name == "round_robin":
        return RoundRobinDriver(tasks)
    raise ValueError(f"Unsupported scheduler driver: {name}")
