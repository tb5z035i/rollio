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
        self.snapshot_poll_count = 0
        self.snapshot_last_loop_us = 0.0
        self.snapshot_avg_loop_us = 0.0
        self.snapshot_max_loop_us = 0.0
        self.snapshot_payload_last_bytes = 0
        self.snapshot_payload_avg_bytes = 0.0
        self.snapshot_payload_max_bytes = 0
        self.render_gap_history_ms: deque[float] = deque(maxlen=self._history_limit)
        self.snapshot_poll_history_ms: deque[float] = deque(maxlen=self._history_limit)
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
        self.snapshot_poll_count = 0
        self.snapshot_last_loop_us = 0.0
        self.snapshot_avg_loop_us = 0.0
        self.snapshot_max_loop_us = 0.0
        self.snapshot_payload_last_bytes = 0
        self.snapshot_payload_avg_bytes = 0.0
        self.snapshot_payload_max_bytes = 0
        self.render_gap_history_ms.clear()
        self.snapshot_poll_history_ms.clear()
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
        *,
        max_frame_width: int | None = None,
        max_frame_height: int | None = None,
    ) -> tuple[float, RuntimeSnapshot]:
        frame_started_at = time.monotonic()
        frame_dt = frame_started_at - self._prev_frame_started_at
        self._prev_frame_started_at = frame_started_at
        if frame_dt > 0:
            self.actual_fps = 0.9 * self.actual_fps + 0.1 / frame_dt
            self.render_gap_history_ms.append(frame_dt * 1000.0)
        snapshot_started_at = time.monotonic()
        try:
            snapshot = runtime.snapshot(
                max_frame_width=max_frame_width,
                max_frame_height=max_frame_height,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            snapshot = runtime.snapshot()
        snapshot_elapsed_us = max(
            0.0,
            (time.monotonic() - snapshot_started_at) * 1_000_000.0,
        )
        self._observe_snapshot_poll(
            snapshot_elapsed_us,
            self._snapshot_payload_bytes(snapshot),
        )
        return frame_started_at, snapshot

    def note_render_work(self, frame_started_at: float) -> None:
        total_elapsed_us = max(0.0, (time.monotonic() - frame_started_at) * 1_000_000.0)
        elapsed_us = max(0.0, total_elapsed_us - self.snapshot_last_loop_us)
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
        timing_lines = build_timing_panel_lines(
            panel_w=panel_w,
            panel_h=max(panel_h - 1, 1),
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
        summary_line = self._build_payload_summary_line(snapshot, panel_w)
        if not summary_line:
            return timing_lines
        return ([timing_lines[0], summary_line, *timing_lines[1:]] + [""] * panel_h)[
            :panel_h
        ]

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

    def _observe_snapshot_poll(
        self,
        elapsed_us: float,
        payload_bytes: int,
    ) -> None:
        self.snapshot_poll_count += 1
        self.snapshot_poll_history_ms.append(elapsed_us / 1000.0)
        self.snapshot_last_loop_us = elapsed_us
        self.snapshot_max_loop_us = max(self.snapshot_max_loop_us, elapsed_us)
        self.snapshot_payload_last_bytes = max(int(payload_bytes), 0)
        self.snapshot_payload_max_bytes = max(
            self.snapshot_payload_max_bytes,
            self.snapshot_payload_last_bytes,
        )
        if self.snapshot_poll_count == 1:
            self.snapshot_avg_loop_us = elapsed_us
            self.snapshot_payload_avg_bytes = float(self.snapshot_payload_last_bytes)
            return
        prev = self.snapshot_poll_count - 1
        self.snapshot_avg_loop_us = (
            (self.snapshot_avg_loop_us * prev) + elapsed_us
        ) / self.snapshot_poll_count
        self.snapshot_payload_avg_bytes = (
            (self.snapshot_payload_avg_bytes * prev) + self.snapshot_payload_last_bytes
        ) / self.snapshot_poll_count

    def _snapshot_payload_bytes(self, snapshot: RuntimeSnapshot) -> int:
        frame_bytes = sum(
            int(getattr(frame, "nbytes", 0))
            for frame in snapshot.latest_frames.values()
            if frame is not None
        )
        state_bytes = sum(
            int(getattr(value, "nbytes", 0))
            for state in snapshot.latest_robot_states.values()
            for value in state.values()
        )
        return frame_bytes + state_bytes

    def _build_payload_summary_line(
        self,
        snapshot: RuntimeSnapshot | None,
        width: int,
    ) -> str:
        if self.snapshot_last_loop_us <= 0.0 and snapshot is None:
            return ""
        summary = (
            f"snapshot {self.snapshot_last_loop_us / 1000.0:0.1f}ms"
            f" avg {self.snapshot_avg_loop_us / 1000.0:0.1f}ms"
            f" | payload {self.snapshot_payload_last_bytes / (1024 * 1024):0.2f}MB"
            f" avg {self.snapshot_payload_avg_bytes / (1024 * 1024):0.2f}MB"
        )
        camera_tasks = (
            snapshot.scheduler_metrics.get("camera_tasks")
            if snapshot is not None
            else None
        )
        if isinstance(camera_tasks, dict) and camera_tasks:
            worst_name, worst_metric = max(
                camera_tasks.items(),
                key=lambda item: (
                    getattr(item[1], "last_backlog", 0),
                    getattr(item[1], "last_step_ms", 0.0),
                    getattr(item[1], "last_copied_bytes", 0),
                ),
            )
            summary += (
                f" | cam {str(worst_name)[:8]}"
                f" s{getattr(worst_metric, 'last_step_ms', 0.0):0.1f}ms"
                f" q{getattr(worst_metric, 'last_backlog', 0)}"
                f" c{getattr(worst_metric, 'last_copied_bytes', 0) / (1024 * 1024):0.2f}MB"
            )
        return summary[: max(width, 1)]
