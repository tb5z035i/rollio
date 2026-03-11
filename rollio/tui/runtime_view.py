"""Shared runtime snapshot polling and render metrics for TUI screens."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from rollio.collect import CollectionRuntimeService, RuntimeSnapshot
from rollio.tui.timing import build_timing_panel_lines, make_timing_trace


@dataclass(frozen=True)
class TaskRateSummary:
    """Aggregated scheduling stats for one task family."""

    actual_hz: float | None
    overruns: int
    avg_step_ms: float | None


@dataclass(frozen=True)
class RuntimeDriverSummary:
    """High-level scheduler summary for one runtime snapshot."""

    telemetry: TaskRateSummary
    control: TaskRateSummary
    driver_last_loop_us: float | None
    driver_avg_loop_us: float | None


class RuntimeViewMonitor:
    """Track render cadence and summarize one worker-backed runtime."""

    def __init__(self, *, history_limit: int = 64) -> None:
        self._history_limit = max(int(history_limit), 8)
        self._runtime_started_at: float | None = None
        self._prev_frame_started_at = time.monotonic()
        self.actual_fps = 0.0
        self.render_loop_count = 0
        self.render_last_loop_us = 0.0
        self.render_avg_loop_us = 0.0
        self.render_gap_history_ms: deque[float] = deque(maxlen=self._history_limit)
        self.render_work_history_ms: deque[float] = deque(maxlen=self._history_limit)
        self.reset()

    def reset(self) -> None:
        now = time.monotonic()
        self._runtime_started_at = None
        self._prev_frame_started_at = now
        self.actual_fps = 0.0
        self.render_loop_count = 0
        self.render_last_loop_us = 0.0
        self.render_avg_loop_us = 0.0
        self.render_gap_history_ms.clear()
        self.render_work_history_ms.clear()

    def mark_runtime_started(self, started_at: float | None = None) -> None:
        self.reset()
        self._runtime_started_at = (
            time.monotonic() if started_at is None else float(started_at)
        )
        self._prev_frame_started_at = self._runtime_started_at

    def poll_snapshot(
        self,
        runtime: CollectionRuntimeService,
    ) -> tuple[float, RuntimeSnapshot]:
        frame_started_at = time.monotonic()
        frame_dt = frame_started_at - self._prev_frame_started_at
        self._prev_frame_started_at = frame_started_at
        if frame_dt > 0:
            self.actual_fps = 0.9 * self.actual_fps + 0.1 / frame_dt
            self.render_gap_history_ms.append(frame_dt * 1000.0)
        return frame_started_at, runtime.snapshot()

    def note_render_work(self, frame_started_at: float) -> None:
        elapsed_us = max(0.0, (time.monotonic() - frame_started_at) * 1_000_000.0)
        self.render_work_history_ms.append(elapsed_us / 1000.0)
        self.render_loop_count += 1
        self.render_last_loop_us = elapsed_us
        if self.render_loop_count == 1:
            self.render_avg_loop_us = elapsed_us
        else:
            prev = self.render_loop_count - 1
            self.render_avg_loop_us = (
                (self.render_avg_loop_us * prev) + elapsed_us
            ) / self.render_loop_count

    def build_timing_lines(
        self,
        *,
        panel_w: int,
        panel_h: int,
        snapshot: RuntimeSnapshot | None,
        target_render_ms: float,
    ) -> list[str]:
        diagnostics = snapshot.timing_diagnostics if snapshot is not None else None
        return build_timing_panel_lines(
            panel_w=panel_w,
            panel_h=panel_h,
            diagnostics=diagnostics,
            render_gap_trace=make_timing_trace(
                tuple(self.render_gap_history_ms),
                target_interval_ms=target_render_ms,
                age_ms=0.0 if self.render_gap_history_ms else None,
            ),
            render_work_trace=make_timing_trace(
                tuple(self.render_work_history_ms),
                target_interval_ms=target_render_ms,
            ),
        )

    def summarize_driver(
        self, snapshot: RuntimeSnapshot | None
    ) -> RuntimeDriverSummary:
        driver_metrics = (
            snapshot.scheduler_metrics.get("driver") if snapshot is not None else None
        )
        task_metrics = driver_metrics.task_metrics if driver_metrics is not None else {}
        return RuntimeDriverSummary(
            telemetry=self._aggregate_prefix(task_metrics, "robot-"),
            control=self._aggregate_prefix(task_metrics, "teleop-"),
            driver_last_loop_us=(
                driver_metrics.last_loop_us
                if driver_metrics is not None and driver_metrics.loop_run_count > 0
                else None
            ),
            driver_avg_loop_us=(
                driver_metrics.avg_loop_us
                if driver_metrics is not None and driver_metrics.loop_run_count > 0
                else None
            ),
        )

    def _aggregate_prefix(
        self,
        task_metrics: dict[str, Any],
        prefix: str,
    ) -> TaskRateSummary:
        runtime_started_at = self._runtime_started_at
        if runtime_started_at is None:
            return TaskRateSummary(actual_hz=None, overruns=0, avg_step_ms=None)
        runtime_age = max(time.monotonic() - runtime_started_at, 1e-6)
        matching = [
            metric for name, metric in task_metrics.items() if name.startswith(prefix)
        ]
        if not matching:
            return TaskRateSummary(actual_hz=None, overruns=0, avg_step_ms=None)
        actual_hz = sum(metric.run_count / runtime_age for metric in matching) / len(
            matching
        )
        overruns = sum(metric.overrun_count for metric in matching)
        avg_step_ms = sum(metric.avg_step_ms for metric in matching) / len(matching)
        return TaskRateSummary(
            actual_hz=actual_hz,
            overruns=overruns,
            avg_step_ms=avg_step_ms,
        )
