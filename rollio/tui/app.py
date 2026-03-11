"""Main TUI application for data collection."""

from __future__ import annotations

import os
import re
import select
import signal
import sys
import termios
import time
import tty
from collections import deque
from typing import Protocol

import numpy as np

from rollio.collect import (
    CollectionRuntimeService,
    RecordedEpisodeSummary,
    RuntimeSnapshot,
    create_runtime_service,
)
from rollio.config.schema import RollioConfig
from rollio.defaults import DEFAULT_CONTROL_INTERVAL_MS
from rollio.tui.renderer import (
    RENDER_MODES,
    MODE_LABELS,
    blit_frame,
    calc_render_size,
    render_frame,
)
from rollio.tui.timing import build_timing_panel_lines, make_timing_trace

# ── Synchronised output ───────────────────────────────────────────────
_SYNC_S = b"\x1b[?2026h"
_SYNC_E = b"\x1b[?2026l"
_ENTER_KEYS = frozenset({"\n", "\r"})
_BACKSPACE_KEYS = frozenset({"\x7f", "\x08"})
_AIRBOT_EEF_DISPLAY_MAX_M = 0.07
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STATUS_TEXT_FG = "\x1b[38;5;250m"


class _EpisodeSummaryLike(Protocol):
    episode_index: int
    duration: float


# ═══════════════════════════════════════════════════════════════════════
#  Terminal helper
# ═══════════════════════════════════════════════════════════════════════


class _Term:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.orig = None
        self.cols = 80
        self.rows = 24

    def __enter__(self):
        self.orig = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        sys.stdout.buffer.write(b"\x1b[?1049h\x1b[?25l")
        sys.stdout.buffer.flush()
        self._resize()
        signal.signal(signal.SIGWINCH, lambda *_: self._resize())
        return self

    def __exit__(self, *_):
        sys.stdout.buffer.write(b"\x1b[?25h\x1b[?1049l")
        sys.stdout.buffer.flush()
        if self.orig:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.orig)

    def _resize(self):
        try:
            cols, rows = os.get_terminal_size()
        except OSError:
            cols, rows = 80, 24
        self.cols = max(cols, 40)
        self.rows = max(rows, 10)

    def key(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Panel helpers
# ═══════════════════════════════════════════════════════════════════════


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _fit_ansi(text: str, width: int) -> str:
    if width <= 0:
        return ""

    out: list[str] = []
    visible = 0
    idx = 0
    while idx < len(text) and visible < width:
        match = _ANSI_RE.match(text, idx)
        if match is not None:
            out.append(match.group(0))
            idx = match.end()
            continue
        out.append(text[idx])
        idx += 1
        visible += 1

    while idx < len(text):
        match = _ANSI_RE.match(text, idx)
        if match is None:
            break
        out.append(match.group(0))
        idx = match.end()

    out.append(" " * max(0, width - visible))
    return "".join(out)


def _key_label(binding: str) -> str:
    normalized = str(binding).strip().lower()
    if binding == " " or normalized in {"space", "spacebar"}:
        return "SPACE"
    if binding in _ENTER_KEYS or normalized in {"enter", "return"}:
        return "ENTER"
    if binding in _BACKSPACE_KEYS or normalized in {"backspace", "bs"}:
        return "BACKSPACE"
    return str(binding)


def _matches_key_binding(key: str | None, binding: str) -> bool:
    if key is None:
        return False
    normalized = str(binding).strip().lower()
    if binding == " " or normalized in {"space", "spacebar"}:
        return key == " "
    if binding in _ENTER_KEYS or normalized in {"enter", "return"}:
        return key in _ENTER_KEYS
    if binding in _BACKSPACE_KEYS or normalized in {"backspace", "bs"}:
        return key in _BACKSPACE_KEYS
    return key == binding


def _review_shortcut_label(primary: str, configured: str) -> str:
    configured_label = _key_label(configured)
    if configured_label == primary:
        return primary
    return f"{primary}/{configured_label}"


def _matches_keep_review(key: str | None, configured: str) -> bool:
    return key in _ENTER_KEYS or _matches_key_binding(key, configured)


def _matches_discard_review(key: str | None, configured: str) -> bool:
    return key in _BACKSPACE_KEYS or _matches_key_binding(key, configured)


def _write_lines(
    out: bytearray,
    *,
    row: int,
    col: int,
    width: int,
    height: int,
    lines: list[str],
) -> None:
    for offset in range(height):
        line = lines[offset] if offset < len(lines) else ""
        padded = _fit_ansi(line, width)
        out.extend(f"\x1b[{row + offset};{col}H".encode())
        out.extend(padded.encode())
        out.extend(b"\x1b[0m")


def _robot_panel_lines(
    states: dict[str, dict[str, np.ndarray] | None],
    robot_types: dict[str, str],
    panel_w: int,
    panel_h: int,
) -> list[str]:
    """Render robot state as coloured ANSI text lines."""
    lines: list[str] = []
    title = f"\x1b[1;93m{' ROBOT STATE ':─<{max(panel_w - 1, 1)}}\x1b[0m"
    lines.append(title)
    for name, st in states.items():
        lines.append(f"\x1b[1;97m {name} \x1b[0m")
        if st is None:
            lines.append("  (no data)")
            continue
        pos = st.get("position")
        if pos is not None:
            robot_type = robot_types.get(name, "")
            for j, p in enumerate(pos):
                value_text, frac = _format_joint_preview(robot_type, float(p))
                bar_len = int(frac * max(panel_w - 18, 0))
                bar = "█" * max(bar_len, 0) + "░" * max(panel_w - 18 - bar_len, 0)
                lines.append(
                    f"  j{j} \x1b[36m{value_text}\x1b[0m " f"\x1b[33m{bar}\x1b[0m"
                )
        control_interval = st.get("control_loop_interval_ms")
        control_target = st.get("control_loop_target_interval_ms")
        if control_interval is not None and len(control_interval) > 0:
            interval_text, frac = _format_control_interval_preview(
                float(control_interval[0]),
                (
                    float(control_target[0])
                    if control_target is not None and len(control_target) > 0
                    else DEFAULT_CONTROL_INTERVAL_MS
                ),
            )
            bar_len = int(frac * max(panel_w - 18, 0))
            bar = "█" * max(bar_len, 0) + "░" * max(panel_w - 18 - bar_len, 0)
            lines.append(
                f"  ctrl \x1b[36m{interval_text}\x1b[0m " f"\x1b[33m{bar}\x1b[0m"
            )
        lines.append("")

    return (lines + [""] * panel_h)[:panel_h]


def _format_joint_preview(robot_type: str, value: float) -> tuple[str, float]:
    if robot_type in {"airbot_e2b", "airbot_g2"}:
        frac = float(np.clip(value / _AIRBOT_EEF_DISPLAY_MAX_M, 0.0, 1.0))
        return f"{value * 1000.0:6.1f}mm", frac
    frac = float(np.clip((value + 2.0) / 4.0, 0.0, 1.0))
    return f"{value:+6.2f}", frac


def _format_control_interval_preview(
    interval_ms: float, target_interval_ms: float
) -> tuple[str, float]:
    target = max(float(target_interval_ms), 1e-6)
    observed = max(float(interval_ms), 0.0)
    frac = float(np.clip(target / max(observed, target), 0.0, 1.0))
    return f"{observed:5.1f}ms", frac


def _estimate_robot_panel_height(
    states: dict[str, dict[str, np.ndarray] | None],
) -> int:
    height = 1
    for state in states.values():
        pos = state.get("position") if state is not None else None
        joints = len(pos) if pos is not None and len(pos) > 0 else 1
        has_control = (
            state is not None
            and state.get("control_loop_interval_ms") is not None
            and len(state.get("control_loop_interval_ms")) > 0
        )
        height += 1 + joints + (1 if has_control else 0) + 1
    return height


def _help_panel_lines(
    cfg: RollioConfig,
    runtime: CollectionRuntimeService,
    panel_w: int,
    panel_h: int,
) -> list[str]:
    """Render the fixed help panel shown on the right side."""
    start_key = _key_label(cfg.controls.start_stop)
    keep_key = _review_shortcut_label("ENTER", cfg.controls.keep)
    discard_key = _review_shortcut_label("BACKSPACE", cfg.controls.discard)
    lines = [
        f"\x1b[1;96m{' HELP ':─<{max(panel_w - 1, 1)}}\x1b[0m",
        f"\x1b[1mMode:\x1b[0m {cfg.mode}",
        f"\x1b[1mRGB codec:\x1b[0m {runtime.video_codec}",
        f"\x1b[1mDepth codec:\x1b[0m {runtime.depth_codec}",
        "",
        "\x1b[1mKeys\x1b[0m",
        f"  {start_key:<8} start / stop",
        f"  {keep_key:<8} keep reviewed",
        f"  {discard_key:<8} discard reviewed",
        "  m        render mode",
        "  \\        debug",
        "  q        quit",
        "",
        "\x1b[1mWorkflow\x1b[0m",
        "  1. start episode",
        "  2. stop episode",
        "  3. keep or discard",
    ]
    return (lines + [""] * panel_h)[:panel_h]


def _state_line(
    snapshot: RuntimeSnapshot,
    pending_episode: _EpisodeSummaryLike | None,
) -> str:
    if snapshot.recording:
        marker = "\x1b[91m●" if int(time.monotonic()) % 2 == 0 else "\x1b[90m○"
        return f"{marker}{_STATUS_TEXT_FG} RECORDING {snapshot.elapsed:.1f}s"
    if pending_episode is not None:
        return (
            f"\x1b[93m?{_STATUS_TEXT_FG} review pending ep#{pending_episode.episode_index} "
            f"{pending_episode.duration:.1f}s"
        )
    return f"\x1b[92m●{_STATUS_TEXT_FG} IDLE"


def _status_lines(
    runtime: CollectionRuntimeService,
    snapshot: RuntimeSnapshot,
    pending_episode: _EpisodeSummaryLike | None,
    *,
    episodes_kept: int,
    pending_exports: int,
    completed_exports: int,
    actual_fps: float,
    render_mode: str,
) -> tuple[str, str]:
    """Build the two bottom status lines."""
    line1 = (
        f" State: {_state_line(snapshot, pending_episode)}"
        f" │ Episodes kept: {episodes_kept}"
        f" │ Export queue: {pending_exports} pending / {completed_exports} done"
        f" │ FPS: {actual_fps:.1f}"
    )
    line2 = (
        f" Render: {MODE_LABELS[render_mode]}"
        f" │ RGB codec: {runtime.video_codec}"
        f" │ Depth codec: {runtime.depth_codec}"
    )
    return line1, line2


# ═══════════════════════════════════════════════════════════════════════
#  Main collection loop
# ═══════════════════════════════════════════════════════════════════════


def run_collection(cfg: RollioConfig) -> None:
    """Run the data collection TUI."""
    runtime = create_runtime_service(cfg, use_worker=True)
    robot_types = {robot.name: robot.type for robot in cfg.robots}
    runtime_opened = False

    episodes_kept = 0
    pending_episode: RecordedEpisodeSummary | None = None
    mode_idx = 0  # start at "true" (24-bit)
    _t_prev_frame = time.monotonic()
    actual_fps = 0.0
    show_debug = False
    render_gap_history_ms: deque[float] = deque(maxlen=64)
    render_work_history_ms: deque[float] = deque(maxlen=64)

    try:
        runtime.open()
        runtime_opened = True
        with _Term() as term:
            out = sys.stdout.buffer
            target_dt = 1.0 / max(cfg.fps, 1)
            while True:
                t0 = time.monotonic()
                _frame_dt = t0 - _t_prev_frame
                _t_prev_frame = t0
                if _frame_dt > 0:
                    actual_fps = 0.9 * actual_fps + 0.1 / _frame_dt
                    render_gap_history_ms.append(_frame_dt * 1000.0)
                render_mode = RENDER_MODES[mode_idx]
                snapshot = runtime.snapshot()

                # ── Input ────────────────────────────────────────
                key = term.key()
                if key == "q":
                    break
                elif key == "m":
                    mode_idx = (mode_idx + 1) % len(RENDER_MODES)
                elif key == "\\":
                    show_debug = not show_debug
                elif _matches_key_binding(key, cfg.controls.start_stop):
                    if snapshot.recording:
                        pending_episode = runtime.stop_episode()
                    else:
                        runtime.start_episode()
                        pending_episode = None
                    snapshot = runtime.snapshot()
                elif pending_episode is not None and _matches_keep_review(
                    key,
                    cfg.controls.keep,
                ):
                    runtime.keep_episode()
                    episodes_kept += 1
                    pending_episode = None
                    snapshot = runtime.snapshot()
                elif pending_episode is not None and _matches_discard_review(
                    key,
                    cfg.controls.discard,
                ):
                    runtime.discard_episode()
                    pending_episode = None
                    snapshot = runtime.snapshot()

                pending_exports, completed_exports = snapshot.export_status

                # ── Read runtime caches ───────────────────────────
                latest_frames = snapshot.latest_frames
                robot_display = snapshot.latest_robot_states

                # ── Layout ───────────────────────────────────────
                W, H = term.cols, term.rows
                status_h = 2
                side_w = (
                    max(32, min(48, W // 3))
                    if show_debug
                    else max(26, min(36, W // 4))
                )
                left_w = max(20, W - side_w)
                body_h = max(2, H - status_h)
                robot_h = (
                    min(
                        max(6, _estimate_robot_panel_height(robot_display)),
                        max(6, body_h // 3),
                    )
                    if robot_display
                    else 0
                )
                cam_h = max(2, body_h - robot_h)

                # ── Compose frame ────────────────────────────────
                cam_names = list(latest_frames.keys())
                robot_lines = (
                    _robot_panel_lines(robot_display, robot_types, left_w, robot_h)
                    if robot_h
                    else []
                )
                timing_diagnostics = snapshot.timing_diagnostics
                side_lines = (
                    build_timing_panel_lines(
                        panel_w=side_w,
                        panel_h=body_h,
                        diagnostics=timing_diagnostics,
                        render_gap_trace=make_timing_trace(
                            tuple(render_gap_history_ms),
                            target_interval_ms=target_dt * 1000.0,
                            age_ms=0.0 if render_gap_history_ms else None,
                        ),
                        render_work_trace=make_timing_trace(
                            tuple(render_work_history_ms),
                            target_interval_ms=target_dt * 1000.0,
                        ),
                    )
                    if show_debug
                    else _help_panel_lines(cfg, runtime, side_w, body_h)
                )
                status_line_1, status_line_2 = _status_lines(
                    runtime,
                    snapshot,
                    pending_episode,
                    episodes_kept=episodes_kept,
                    pending_exports=pending_exports,
                    completed_exports=completed_exports,
                    actual_fps=actual_fps,
                    render_mode=render_mode,
                )

                frame_out = bytearray()
                frame_out.extend(b"\x1b[H")  # cursor home
                _write_lines(
                    frame_out,
                    row=1,
                    col=1,
                    width=left_w,
                    height=body_h,
                    lines=[],
                )

                # Camera(s) on the left side — aspect preserved
                if cam_names:
                    cam_count = len(cam_names)
                    cam_row = 1
                    for idx, cn in enumerate(cam_names):
                        remaining = max(cam_h - (cam_row - 1), 1)
                        remaining_cams = cam_count - idx
                        cam_h_each = max(1, remaining // max(remaining_cams, 1))
                        dedicated_label_row = cam_h_each > 1
                        preview_h = max(1, cam_h_each - 1) if dedicated_label_row else 1
                        rendered_preview_h = 1
                        frame = latest_frames.get(cn)
                        if frame is not None:
                            fh, fw = frame.shape[:2]
                            rw, rh = calc_render_size(fw, fh, left_w, preview_h)
                            rendered_preview_h = max(1, min(preview_h, rh))
                            rendered = render_frame(frame, rw, rh, render_mode)
                            frame_out.extend(blit_frame(rendered, cam_row, 1))
                        else:
                            _write_lines(
                                frame_out,
                                row=cam_row,
                                col=1,
                                width=left_w,
                                height=preview_h,
                                lines=["\x1b[90m(no preview)\x1b[0m"],
                            )
                        label_row = cam_row if not dedicated_label_row else cam_row + rendered_preview_h
                        _write_lines(
                            frame_out,
                            row=label_row,
                            col=1,
                            width=left_w,
                            height=1,
                            lines=[f"\x1b[1;96m[{idx + 1}] {cn}\x1b[0m"],
                        )
                        cam_row += cam_h_each
                else:
                    _write_lines(
                        frame_out,
                        row=1,
                        col=1,
                        width=left_w,
                        height=cam_h,
                        lines=["\x1b[90m(no camera previews)\x1b[0m"],
                    )

                # Robot panel — lower-left window
                if robot_h:
                    _write_lines(
                        frame_out,
                        row=cam_h + 1,
                        col=1,
                        width=left_w,
                        height=robot_h,
                        lines=robot_lines,
                    )

                # Right panel — fixed help only
                _write_lines(
                    frame_out,
                    row=1,
                    col=left_w + 1,
                    width=side_w,
                    height=body_h,
                    lines=side_lines,
                )

                # ── Status bar ───────────────────────────────────
                status_bytes = (
                    f"\x1b[{H-1};1H"
                    f"\x1b[48;5;236m\x1b[38;5;250m"
                    f"{_fit_ansi(status_line_1, W)}"
                    f"\x1b[{H};1H"
                    f"\x1b[48;5;236m\x1b[38;5;250m"
                    f"{_fit_ansi(status_line_2, W)}"
                    f"\x1b[0m"
                ).encode()

                out.write(_SYNC_S + bytes(frame_out) + status_bytes + _SYNC_E)
                out.flush()
                render_work_history_ms.append((time.monotonic() - t0) * 1000.0)

                # ── Throttle ─────────────────────────────────────
                dt = time.monotonic() - t0
                if dt < target_dt:
                    time.sleep(target_dt - dt)
    finally:
        if runtime_opened:
            try:
                print("Returning robots to zero...", flush=True)
                runtime.return_robots_to_zero(timeout=5.0)
            finally:
                runtime.close()
