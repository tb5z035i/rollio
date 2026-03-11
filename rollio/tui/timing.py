"""Shared helpers for timing-oriented TUI debug panels."""

from __future__ import annotations

from collections.abc import Sequence

from rollio.collect.runtime import RuntimeTimingDiagnostics, TimingTrace

_TIMELINE_CHARS = ".-=+*!"


def make_timing_trace(
    intervals_ms: Sequence[float],
    *,
    target_interval_ms: float | None = None,
    age_ms: float | None = None,
) -> TimingTrace:
    """Build one compact timing trace from recent interval samples."""
    values = tuple(max(0.0, float(value)) for value in intervals_ms)
    return TimingTrace(
        intervals_ms=values,
        target_interval_ms=target_interval_ms,
        last_gap_ms=values[-1] if values else None,
        max_gap_ms=max(values) if values else None,
        age_ms=age_ms,
    )


def build_timing_panel_lines(
    *,
    panel_w: int,
    panel_h: int,
    diagnostics: RuntimeTimingDiagnostics | None,
    render_gap_trace: TimingTrace | None = None,
    render_work_trace: TimingTrace | None = None,
) -> list[str]:
    """Render a compact numeric panel plus interval timelines."""
    width = max(panel_w, 16)
    lines = [f"\x1b[1;95m{' TIMING ':-<{max(width - 1, 1)}}\x1b[0m"]
    max_traces = max(1, (panel_h - len(lines)) // 2)

    traces: list[tuple[str, TimingTrace]] = []
    if render_gap_trace is not None:
        traces.append(("render_gap", render_gap_trace))
    if diagnostics is not None:
        traces.append(("sched", diagnostics.scheduler_loop))
        traces.extend(
            (
                _compact_label(name, prefix="robot"),
                trace,
            )
            for name, trace in _select_robot_traces(
                diagnostics.valid_robot_samples,
                max(1, max_traces),
            )
        )
        traces.extend(
            [
                ("ctrl", diagnostics.control_runs),
                ("telem", diagnostics.telemetry_runs),
            ]
        )
    if render_work_trace is not None:
        traces.append(("render_work", render_work_trace))
    traces = traces[:max_traces]

    if len(traces) == 0:
        lines.append("\x1b[90m(no timing diagnostics)\x1b[0m")
        return (lines + [""] * panel_h)[:panel_h]

    for label, trace in traces:
        lines.append(_trace_summary_line(label, trace, width))
        lines.append(_trace_timeline_line(trace, width))

    return (lines + [""] * panel_h)[:panel_h]


def _select_robot_traces(
    traces: dict[str, TimingTrace],
    max_items: int,
) -> list[tuple[str, TimingTrace]]:
    if max_items <= 0:
        return []
    ordered = sorted(
        traces.items(),
        key=lambda item: (
            item[1].last_gap_ms if item[1].last_gap_ms is not None else -1.0,
            item[0],
        ),
        reverse=True,
    )
    return ordered[:max_items]


def _compact_label(label: str, *, prefix: str = "") -> str:
    if prefix and label.startswith(prefix):
        label = label[len(prefix) :]
    cleaned = str(label).strip(" _-")
    if not cleaned:
        return prefix or "trace"
    return cleaned[:12]


def _format_metric(value_ms: float | None, *, fallback: str = "n/a") -> str:
    if value_ms is None:
        return fallback
    return f"{value_ms:4.1f}"


def _trace_summary_line(label: str, trace: TimingTrace, width: int) -> str:
    target_text = (
        f" t{trace.target_interval_ms:4.1f}"
        if trace.target_interval_ms is not None
        else ""
    )
    summary = (
        f"{label:<12} "
        f"l{_format_metric(trace.last_gap_ms)} "
        f"a{_format_metric(trace.age_ms)} "
        f"m{_format_metric(trace.max_gap_ms)}"
        f"{target_text}"
    )
    return summary[:width]


def _trace_timeline_line(trace: TimingTrace, width: int) -> str:
    timeline_w = max(min(width - 2, 32), 8)
    samples = _resample_tail(trace.intervals_ms, timeline_w)
    if len(samples) == 0:
        return "\x1b[90m(no history)\x1b[0m"[:width]
    timeline = "".join(_interval_char(value, trace.target_interval_ms) for value in samples)
    return f"  {timeline}"[:width]


def _resample_tail(values: Sequence[float], width: int) -> tuple[float, ...]:
    if len(values) <= width:
        return tuple(values)
    start = len(values) - width
    return tuple(values[start:])


def _interval_char(value_ms: float, target_interval_ms: float | None) -> str:
    reference = max(float(target_interval_ms or 1.0), 1e-6)
    ratio = max(float(value_ms), 0.0) / reference
    if ratio < 0.75:
        return _TIMELINE_CHARS[0]
    if ratio < 1.25:
        return _TIMELINE_CHARS[1]
    if ratio < 2.0:
        return _TIMELINE_CHARS[2]
    if ratio < 4.0:
        return _TIMELINE_CHARS[3]
    if ratio < 8.0:
        return _TIMELINE_CHARS[4]
    return _TIMELINE_CHARS[5]

