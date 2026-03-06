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

import numpy as np

from rollio.collect import AsyncCollectionRuntime, RecordedEpisode
from rollio.config.schema import RollioConfig
from rollio.tui.renderer import (
    RENDER_MODES, MODE_LABELS, blit_frame, calc_render_size, render_frame,
)

# ── Synchronised output ───────────────────────────────────────────────
_SYNC_S = b"\x1b[?2026h"
_SYNC_E = b"\x1b[?2026l"


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
    return len(re.sub(r"\x1b\[[0-9;]*m", "", text))


def _pad_ansi(text: str, width: int) -> str:
    visible = _visible_len(text)
    return text + " " * max(0, width - visible)


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
        padded = _pad_ansi(line, width)
        out.extend(f"\x1b[{row + offset};{col}H".encode())
        out.extend(padded[:width + 80].encode())
        out.extend(b"\x1b[0m")


def _robot_panel_lines(
    states: dict[str, dict[str, np.ndarray] | None],
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
        vel = st.get("velocity")
        if pos is not None:
            for j, p in enumerate(pos):
                v = vel[j] if vel is not None and j < len(vel) else 0.0
                # Colour bar for position  (-2..+2 range → bar)
                bar_len = int(np.clip((p + 2) / 4, 0, 1) * (panel_w - 18))
                bar = "█" * max(bar_len, 0) + "░" * max(panel_w - 18 - bar_len, 0)
                lines.append(
                    f"  j{j} \x1b[36m{p:+6.2f}\x1b[0m "
                    f"\x1b[33m{bar}\x1b[0m")
        lines.append("")

    return (lines + [""] * panel_h)[:panel_h]


def _help_panel_lines(
    cfg: RollioConfig,
    runtime: AsyncCollectionRuntime,
    panel_w: int,
    panel_h: int,
) -> list[str]:
    """Render the fixed help panel shown on the right side."""
    start_key = "SPACE" if cfg.controls.start_stop == " " else cfg.controls.start_stop
    lines = [
        f"\x1b[1;96m{' HELP ':─<{max(panel_w - 1, 1)}}\x1b[0m",
        f"\x1b[1mMode:\x1b[0m {cfg.mode}",
        f"\x1b[1mRGB codec:\x1b[0m {runtime.video_codec}",
        f"\x1b[1mDepth codec:\x1b[0m {runtime.depth_codec}",
        "",
        "\x1b[1mKeys\x1b[0m",
        f"  {start_key:<8} start / stop",
        f"  {cfg.controls.keep:<8} keep reviewed",
        f"  {cfg.controls.discard:<8} discard reviewed",
        "  m        render mode",
        "  q        quit",
        "",
        "\x1b[1mWorkflow\x1b[0m",
        "  1. start episode",
        "  2. stop episode",
        "  3. keep or discard",
    ]
    return (lines + [""] * panel_h)[:panel_h]


def _state_line(runtime: AsyncCollectionRuntime, pending_episode: RecordedEpisode | None) -> str:
    if runtime.recording:
        return f"REC {runtime.elapsed:.1f}s"
    if pending_episode is not None:
        return (
            f"REVIEW ep#{pending_episode.episode_index} "
            f"{pending_episode.duration:.1f}s"
        )
    return "IDLE"


def _status_lines(
    runtime: AsyncCollectionRuntime,
    pending_episode: RecordedEpisode | None,
    *,
    episodes_kept: int,
    pending_exports: int,
    completed_exports: int,
    actual_fps: float,
    render_mode: str,
) -> tuple[str, str]:
    """Build the two bottom status lines."""
    line1 = (
        f" State: {_state_line(runtime, pending_episode)}"
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
    runtime = AsyncCollectionRuntime.from_config(cfg)

    episodes_kept = 0
    pending_episode: RecordedEpisode | None = None
    mode_idx = 1   # start at "16" (lower bandwidth)
    _t_prev_frame = time.monotonic()
    actual_fps = 0.0
    runtime.open()

    try:
        with _Term() as term:
            out = sys.stdout.buffer
            target_dt = 1.0 / max(cfg.fps, 1)
            while True:
                t0 = time.monotonic()
                _frame_dt = t0 - _t_prev_frame
                _t_prev_frame = t0
                if _frame_dt > 0:
                    actual_fps = 0.9 * actual_fps + 0.1 / _frame_dt
                render_mode = RENDER_MODES[mode_idx]

                # ── Input ────────────────────────────────────────
                key = term.key()
                if key == "q":
                    break
                elif key == "m":
                    mode_idx = (mode_idx + 1) % len(RENDER_MODES)
                elif key == cfg.controls.start_stop:
                    if runtime.recording:
                        pending_episode = runtime.stop_episode()
                    else:
                        runtime.start_episode()
                elif key == cfg.controls.keep and pending_episode is not None:
                    runtime.keep_episode(pending_episode)
                    episodes_kept += 1
                    pending_episode = None
                elif key == cfg.controls.discard and pending_episode is not None:
                    runtime.discard_episode(pending_episode)
                    pending_episode = None

                pending_exports, completed_exports = runtime.export_status()

                # ── Read runtime caches ───────────────────────────
                latest_frames = runtime.latest_frames()
                robot_display = runtime.latest_robot_states()

                # ── Layout ───────────────────────────────────────
                W, H = term.cols, term.rows
                status_h = 2
                help_w = max(26, min(36, W // 4))
                left_w = max(20, W - help_w)
                body_h = max(2, H - status_h)
                robot_h = min(max(6, len(robot_display) * 8), max(6, body_h // 3)) if robot_display else 0
                cam_h = max(2, body_h - robot_h)

                # ── Compose frame ────────────────────────────────
                cam_names = list(latest_frames.keys())
                robot_lines = _robot_panel_lines(robot_display, left_w, robot_h) if robot_h else []
                help_lines = _help_panel_lines(cfg, runtime, help_w, body_h)
                status_line_1, status_line_2 = _status_lines(
                    runtime,
                    pending_episode,
                    episodes_kept=episodes_kept,
                    pending_exports=pending_exports,
                    completed_exports=completed_exports,
                    actual_fps=actual_fps,
                    render_mode=render_mode,
                )

                frame_out = bytearray()
                frame_out.extend(b"\x1b[H")   # cursor home
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
                    cam_h_each = max(2, cam_h // max(len(cam_names), 1))
                    cam_row = 1
                    for cn in cam_names:
                        frame = latest_frames.get(cn)
                        if frame is not None:
                            fh, fw = frame.shape[:2]
                            rw, rh = calc_render_size(
                                fw, fh, left_w, cam_h_each)
                            rendered = render_frame(
                                frame, rw, rh, render_mode)
                            frame_out.extend(
                                blit_frame(rendered, cam_row, 1))
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
                    width=help_w,
                    height=body_h,
                    lines=help_lines,
                )

                # ── Status bar ───────────────────────────────────
                status_bytes = (
                    f"\x1b[{H-1};1H"
                    f"\x1b[48;5;236m\x1b[38;5;250m"
                    f"{status_line_1[:W].ljust(W)}"
                    f"\x1b[{H};1H"
                    f"\x1b[48;5;236m\x1b[38;5;250m"
                    f"{status_line_2[:W].ljust(W)}"
                    f"\x1b[0m"
                ).encode()

                out.write(_SYNC_S + bytes(frame_out) + status_bytes + _SYNC_E)
                out.flush()

                # ── Throttle ─────────────────────────────────────
                dt = time.monotonic() - t0
                if dt < target_dt:
                    time.sleep(target_dt - dt)
    finally:
        runtime.close()
