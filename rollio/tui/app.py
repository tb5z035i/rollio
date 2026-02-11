"""Main TUI application for data collection."""
from __future__ import annotations

import os
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

from rollio.config.schema import RollioConfig
from rollio.episode.recorder import EpisodeRecorder, EpisodeData
from rollio.episode.writer import LeRobotV21Writer
from rollio.sensors.base import ImageSensor, RobotSensor
from rollio.sensors.pseudo_camera import PseudoCamera
from rollio.sensors.pseudo_robot import PseudoRobot
from rollio.tui.renderer import render_frame

# ── Synchronised output ───────────────────────────────────────────────
_SYNC_S = b"\x1b[?2026h"
_SYNC_E = b"\x1b[?2026l"


# ═══════════════════════════════════════════════════════════════════════
#  Sensor factory
# ═══════════════════════════════════════════════════════════════════════

def _build_sensors(cfg: RollioConfig) -> tuple[
        dict[str, ImageSensor], dict[str, RobotSensor]]:
    cameras: dict[str, ImageSensor] = {}
    for cc in cfg.cameras:
        if cc.type == "pseudo":
            cameras[cc.name] = PseudoCamera(
                name=cc.name, width=cc.width, height=cc.height, fps=cc.fps)
        else:
            raise NotImplementedError(f"Camera type '{cc.type}' not yet implemented")

    robots: dict[str, RobotSensor] = {}
    for rc in cfg.robots:
        if rc.type == "pseudo":
            robots[rc.name] = PseudoRobot(
                name=rc.name, n_joints=rc.num_joints, role=rc.role)
        else:
            raise NotImplementedError(f"Robot type '{rc.type}' not yet implemented")

    return cameras, robots


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
        self.cols, self.rows = os.get_terminal_size()

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
    cameras, robots = _build_sensors(cfg)

    # Open all sensors
    for c in cameras.values():
        c.open()
    for r in robots.values():
        r.open()

    recorder = EpisodeRecorder(cameras, robots, fps=cfg.fps)
    writer = LeRobotV21Writer(
        root=cfg.storage.root,
        project_name=cfg.project_name,
        fps=cfg.fps,
    )

    episodes_kept = 0
    pending_episode: EpisodeData | None = None

    with _Term() as term:
        out = sys.stdout.buffer
        target_dt = 1.0 / cfg.fps

        try:
            while True:
                t0 = time.monotonic()

                # ── Input ────────────────────────────────────────
                key = term.key()
                if key == "q":
                    break
                elif key == cfg.controls.start_stop:
                    if recorder.recording:
                        pending_episode = recorder.stop()
                    else:
                        recorder.start()
                elif key == cfg.controls.keep and pending_episode is not None:
                    writer.write(pending_episode)
                    episodes_kept += 1
                    pending_episode = None
                elif key == cfg.controls.discard and pending_episode is not None:
                    pending_episode = None

                # ── Read sensors ─────────────────────────────────
                if recorder.recording:
                    latest_frames = recorder.tick()
                    _, latest_states = recorder.peek_sensors()
                    # Use frames from tick for camera, peek for robot display
                    # (tick already recorded robot states)
                    robot_display = {}
                    for name, rob in robots.items():
                        _, st = rob.read()
                        robot_display[name] = st
                else:
                    latest_frames, robot_display = recorder.peek_sensors()

                # ── Layout ───────────────────────────────────────
                W, H = term.cols, term.rows
                status_h = 2
                cam_w = max(10, W * 2 // 3)
                panel_w = max(10, W - cam_w)
                body_h = max(2, H - status_h)

                # ── Render camera(s) ─────────────────────────────
                cam_names = list(latest_frames.keys())
                if cam_names:
                    # Stack multiple cameras vertically
                    cam_h_each = max(2, body_h // max(len(cam_names), 1))
                    cam_bytes = bytearray()
                    for ci, cn in enumerate(cam_names):
                        frame = latest_frames.get(cn)
                        if frame is not None:
                            rendered = render_frame(frame, cam_w, cam_h_each)
                            cam_bytes.extend(rendered)
                            if ci < len(cam_names) - 1:
                                cam_bytes.extend(_RST_NL)
                    cam_bytes.extend(b"\x1b[0m")
                else:
                    cam_bytes = b"(no cameras)"

                # ── Render robot panel ───────────────────────────
                robot_bytes = _robot_panel(robot_display, panel_w, body_h)

                # ── Compose frame ────────────────────────────────
                frame_out = bytearray()
                frame_out.extend(b"\x1b[H")   # cursor home

                # Write camera region line by line, then robot panel
                cam_lines = bytes(cam_bytes).split(b"\x1b[0m\n")
                rob_text = robot_bytes.decode("utf-8", errors="replace")
                rob_lines_raw = rob_text.split("\x1b[0m")
                rob_lines = [r for r in rob_lines_raw if r or True]

                for y in range(body_h):
                    # Camera portion
                    if y < len(cam_lines):
                        frame_out.extend(cam_lines[y])
                    else:
                        frame_out.extend(b" " * cam_w)
                    frame_out.extend(b"\x1b[0m")

                    # Robot panel portion (position cursor)
                    frame_out.extend(
                        f"\x1b[{y+1};{cam_w+1}H".encode())
                    if y < len(rob_lines):
                        frame_out.extend(rob_lines[y].encode())
                    frame_out.extend(b"\x1b[0m\n")

                # ── Status bar ───────────────────────────────────
                if recorder.recording:
                    state_str = (f"\x1b[1;91m● REC  "
                                 f"{recorder.elapsed:.1f}s\x1b[0m")
                elif pending_episode is not None:
                    state_str = (f"\x1b[1;93m■ REVIEW  "
                                 f"ep#{pending_episode.episode_index} "
                                 f"{pending_episode.duration:.1f}s  "
                                 f"[k]=keep [d]=discard\x1b[0m")
                else:
                    state_str = "\x1b[1;92m⏸ IDLE\x1b[0m"

                fps = 1.0 / max(time.monotonic() - t0, 1e-9)
                bar1 = (f" {state_str}  │  "
                        f"Episodes: {episodes_kept}  │  "
                        f"FPS: {fps:.0f}  │  "
                        f"[SPACE]=start/stop  [k]=keep  [d]=discard  "
                        f"[q]=quit")
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
            for c in cameras.values():
                c.close()
            for r in robots.values():
                r.close()

_RST_NL = b"\x1b[0m\n"
