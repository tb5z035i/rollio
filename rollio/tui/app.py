"""Main TUI application for data collection."""
from __future__ import annotations

import os
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
#  Right-panel: robot state
# ═══════════════════════════════════════════════════════════════════════

def _robot_panel(states: dict[str, dict[str, np.ndarray] | None],
                 panel_w: int, panel_h: int) -> bytes:
    """Render robot state as coloured ANSI text block."""
    lines: list[str] = []
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

    # Trim / pad to panel_h
    lines = lines[:panel_h]
    while len(lines) < panel_h:
        lines.append("")

    # Encode with fixed width + cursor positioning
    out = bytearray()
    for i, line in enumerate(lines):
        # Strip ANSI for length calculation (rough)
        import re
        vis_len = len(re.sub(r"\x1b\[[0-9;]*m", "", line))
        padded = line + " " * max(0, panel_w - vis_len)
        out.extend(padded[:panel_w + 40].encode())  # generous for ANSI
        out.extend(b"\x1b[0m")
    return bytes(out)


# ═══════════════════════════════════════════════════════════════════════
#  Main collection loop
# ═══════════════════════════════════════════════════════════════════════

def run_collection(cfg: RollioConfig) -> None:
    """Run the data collection TUI."""
    runtime = AsyncCollectionRuntime.from_config(cfg)

    episodes_kept = 0
    pending_episode: RecordedEpisode | None = None
    mode_idx = 1   # start at "16" (lower bandwidth)
    show_debug = False
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
                elif key == "\\":
                    show_debug = not show_debug
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
                _t_sns = time.monotonic()
                latest_frames = runtime.latest_frames()
                robot_display = runtime.latest_robot_states()
                _t_sns = time.monotonic() - _t_sns

                # ── Layout ───────────────────────────────────────
                _t_rnd = time.monotonic()
                W, H = term.cols, term.rows
                status_h = 2
                panel_w = max(10, W // 3)
                cam_area_w = max(10, W - panel_w)
                body_h = max(2, H - status_h)

                # ── Compose frame ────────────────────────────────
                cam_names = list(latest_frames.keys())
                robot_bytes = _robot_panel(robot_display, panel_w, body_h)

                frame_out = bytearray()
                frame_out.extend(b"\x1b[H")   # cursor home

                # Camera(s) — fixed 480p resolution, aspect preserved
                if cam_names:
                    cam_h_each = max(2, body_h // max(len(cam_names), 1))
                    cam_row = 1
                    for cn in cam_names:
                        frame = latest_frames.get(cn)
                        if frame is not None:
                            fh, fw = frame.shape[:2]
                            rw, rh = calc_render_size(
                                fw, fh, cam_area_w, cam_h_each)
                            rendered = render_frame(
                                frame, rw, rh, render_mode)
                            frame_out.extend(
                                blit_frame(rendered, cam_row, 1))
                        cam_row += cam_h_each

                # Robot panel — right side
                rob_text = robot_bytes.decode("utf-8", errors="replace")
                rob_lines = rob_text.split("\x1b[0m")
                for y in range(body_h):
                    frame_out.extend(
                        f"\x1b[{y+1};{cam_area_w+1}H".encode())
                    if y < len(rob_lines):
                        frame_out.extend(rob_lines[y].encode())
                    frame_out.extend(b"\x1b[0m\n")
                _t_rnd = time.monotonic() - _t_rnd

                # ── Debug overlay ─────────────────────────────
                if show_debug:
                    _dw = 28
                    _dc = max(1, W - _dw)
                    _dbg = [
                        f"\x1b[1m{'─── DEBUG ':─<{_dw}}\x1b[0m",
                        f"  {'FPS:':>8s} {actual_fps:6.1f}",
                        f"  {'Render:':>8s} {_t_rnd*1000:5.1f} ms",
                        f"  {'Sensor:':>8s} {_t_sns*1000:5.1f} ms",
                        f"  {'Total:':>8s} {(time.monotonic()-t0)*1000:5.1f} ms",
                        f"  {'Mode:':>8s} {MODE_LABELS[render_mode]}",
                        f"  {'Output:':>8s} {len(frame_out)/1024:5.1f} kB",
                        f"{'─' * _dw}",
                    ]
                    for _di, _dl in enumerate(_dbg):
                        _st = ("\x1b[48;5;234m\x1b[38;5;82m\x1b[1m"
                               if _di == 0 else
                               "\x1b[48;5;234m\x1b[38;5;250m")
                        frame_out.extend(
                            f"\x1b[{2+_di};{_dc}H"
                            f"{_st}{_dl:<{_dw}}"
                            f"\x1b[0m".encode())

                # ── Status bar ───────────────────────────────────
                if runtime.recording:
                    state_str = (f"\x1b[1;91m● REC  "
                                 f"{runtime.elapsed:.1f}s\x1b[0m")
                elif pending_episode is not None:
                    state_str = (f"\x1b[1;93m■ REVIEW  "
                                 f"ep#{pending_episode.episode_index} "
                                 f"{pending_episode.duration:.1f}s  "
                                 f"[k]=keep [d]=discard\x1b[0m")
                else:
                    state_str = "\x1b[1;92m⏸ IDLE\x1b[0m"

                bar1 = (f" {state_str}  │  "
                        f"Ep: {episodes_kept}  │  "
                        f"Export: {pending_exports} pending/{completed_exports} done  │  "
                        f"FPS: {actual_fps:.0f}  │  "
                        f"{MODE_LABELS[render_mode]}  │  "
                        f"[SPACE]=rec  [k]=keep  [d]=discard  "
                        f"[m]=mode  [\\]=debug  [q]=quit")
                bar1_clean = bar1.replace("\x1b[1;91m", "").replace(
                    "\x1b[1;93m", "").replace(
                    "\x1b[1;92m", "").replace("\x1b[0m", "")

                status_bytes = (
                    f"\x1b[{H-1};1H"
                    f"\x1b[48;5;236m\x1b[38;5;250m"
                    f"{bar1_clean[:W].ljust(W)}"
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
