"""Interactive TUI Setup Wizard.

Scans hardware, shows live camera preview / robot oscillation, and
prompts the user for channel names.  Works without a desktop environment.
"""
from __future__ import annotations

import io
import math
import os
import select
import signal
import sys
import termios
import time
import tty

import cv2
import numpy as np

from rollio.config.schema import (
    CameraConfig, ControlConfig, RobotConfig, RollioConfig, StorageConfig,
)
from rollio.sensors.pseudo_camera import PseudoCamera
from rollio.sensors.pseudo_robot import PseudoRobot
from rollio.sensors.scanner import DetectedDevice, scan_cameras, scan_robots
from rollio.tui.renderer import (
    RENDER_MODES, MODE_LABELS, blit_frame, calc_render_size, render_frame,
)

# ── Sync output ────────────────────────────────────────────────────────
_SY_S = b"\x1b[?2026h"
_SY_E = b"\x1b[?2026l"


# ═══════════════════════════════════════════════════════════════════════
#  Low-level terminal helpers
# ═══════════════════════════════════════════════════════════════════════

class _Term:
    """Raw-mode terminal with key reading and inline text editing."""

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

    def read_key(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            return ch
        return None

    def read_key_blocking(self, timeout: float = 0.05) -> str | None:
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Helper: open a temp sensor for preview
# ═══════════════════════════════════════════════════════════════════════

def _make_camera(dev: DetectedDevice) -> PseudoCamera | None:
    """Instantiate a camera sensor from a DetectedDevice for preview."""
    if dev.dtype == "pseudo":
        cam = PseudoCamera(
            name="preview", width=dev.properties.get("width", 640),
            height=dev.properties.get("height", 480),
            fps=dev.properties.get("fps", 30))
        cam.open()
        return cam
    elif dev.dtype == "v4l2":
        # Reuse PseudoCamera for now; real v4l2 would use cv2.VideoCapture
        # but we don't have V4L2Camera implemented yet — fall back to
        # attempting a real capture for preview
        try:
            cap = cv2.VideoCapture(int(dev.device_id))
            if cap.isOpened():
                # Wrap in a simple adapter
                class _V4L2Preview:
                    def read(self):
                        from rollio.utils.time import monotonic_sec
                        ret, f = cap.read()
                        return monotonic_sec(), f if ret else np.zeros(
                            (480, 640, 3), np.uint8)
                    def close(self): cap.release()
                return _V4L2Preview()  # type: ignore
        except Exception:
            pass
    return None


def _make_robot(dev: DetectedDevice) -> PseudoRobot | None:
    if dev.dtype == "pseudo":
        rob = PseudoRobot(
            name="preview",
            n_joints=dev.properties.get("num_joints", 6))
        rob.open()
        return rob
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Wizard screens
# ═══════════════════════════════════════════════════════════════════════

def _draw_header(out, W: int, step: int, total: int, title: str):
    """Draw wizard header bar."""
    hdr = f"  ROLLIO SETUP WIZARD  —  Step {step}/{total}: {title}  "
    out.write(f"\x1b[1;1H\x1b[48;5;24m\x1b[97;1m{hdr.ljust(W)}\x1b[0m".encode())


def _draw_text(out, row: int, col: int, text: str):
    out.write(f"\x1b[{row};{col}H{text}\x1b[0m".encode())


def _prompt_line(term: _Term, out, row: int, col: int,
                 prompt: str, default: str = "") -> str | None:
    """Inline text editor. Returns entered string, or None on Escape/q."""
    buf = list(default)
    while True:
        # Draw prompt + current text + cursor
        display = "".join(buf)
        line = f"{prompt}\x1b[97;1m{display}\x1b[0m\x1b[5m_\x1b[0m"
        padding = " " * max(0, term.cols - col - len(prompt) - len(display) - 2)
        out.write(f"\x1b[{row};{col}H{line}{padding}".encode())
        out.flush()

        key = term.read_key_blocking(0.05)
        if key is None:
            continue
        if key == "\n" or key == "\r":
            return "".join(buf) if buf else default
        elif key == "\x1b":      # Escape
            return None
        elif key == "\x7f" or key == "\x08":  # Backspace
            if buf:
                buf.pop()
        elif key.isprintable():
            buf.append(key)


def _screen_cameras(term: _Term, out, devices: list[DetectedDevice]
                    ) -> list[CameraConfig]:
    """Camera identification screen — live preview + name prompt."""
    configs: list[CameraConfig] = []
    total_steps = 3

    mode_idx = 1   # start at "16" (lower bandwidth than 256)
    show_debug = False
    _t_prev = time.monotonic()
    _fps = 0.0

    for i, dev in enumerate(devices):
        cam = _make_camera(dev)
        chosen_name: str | None = None
        default_name = f"cam_{i}"
        phase = "preview"          # "preview" → "name" → accepted
        _needs_clear = True

        while chosen_name is None:
            W, H = term.cols, term.rows

            # Build entire frame into a buffer, then write atomically
            # to avoid flicker (same strategy as the collection loop).
            buf = io.BytesIO()

            # Header
            _draw_header(buf, W, 1, total_steps, "Cameras")

            # Device info
            mode = RENDER_MODES[mode_idx]
            _draw_text(buf, 3, 2,
                       f"Camera {i+1}/{len(devices)}: "
                       f"\x1b[96m{dev.label}\x1b[0m  "
                       f"\x1b[90m[{MODE_LABELS[mode]}]\x1b[0m")
            _draw_text(buf, 4, 2,
                       f"Type: {dev.dtype}  Device: {dev.device_id}")

            # Live preview — fixed 480p resolution, aspect preserved
            cam_w = dev.properties.get("width", 640)
            cam_h = dev.properties.get("height", 480)
            preview_y = 6
            avail_h = max(4, H - 14)
            avail_w = W - 4
            preview_w, preview_h = calc_render_size(
                cam_w, cam_h, avail_w, avail_h)
            if cam is not None:
                _, frame = cam.read()
                if frame is not None:
                    rendered = render_frame(
                        frame, preview_w, preview_h, mode)
                    buf.write(blit_frame(rendered, preview_y, 3))
            else:
                _draw_text(buf, preview_y, 3, "(no preview available)")

            prompt_row = preview_y + preview_h + 1

            if phase == "preview":
                # Live-preview phase: camera refreshes every iteration,
                # shortcut keys are handled directly (not inside _prompt_line).
                _draw_text(buf, prompt_row, 2,
                           "\x1b[33m[Enter]\x1b[0m name  "
                           "\x1b[33m[s]\x1b[0m skip  "
                           "\x1b[33m[m]\x1b[0m mode  "
                           "\x1b[33m[\\]\x1b[0m debug  "
                           "\x1b[33m[Esc/q]\x1b[0m quit")

                # FPS tracking
                _t_now = time.monotonic()
                _dt = _t_now - _t_prev
                _t_prev = _t_now
                if _dt > 0:
                    _fps = 0.9 * _fps + 0.1 / _dt

                if show_debug:
                    _draw_text(buf, 2, W - 18,
                               f"\x1b[48;5;234m\x1b[38;5;82m FPS: {_fps:5.1f} \x1b[0m")

                # Erase leftover content below prompt
                buf.write(f"\x1b[{prompt_row + 1};1H\x1b[J".encode())

                _clear = b"\x1b[2J" if _needs_clear else b""
                _needs_clear = False
                out.write(
                    _SY_S + b"\x1b[H" + _clear + buf.getvalue() + _SY_E)
                out.flush()

                key = term.read_key()
                if key == "\n" or key == "\r":
                    phase = "name"
                    _needs_clear = True
                elif key == "s":
                    break               # skip this camera
                elif key == "m":
                    mode_idx = (mode_idx + 1) % len(RENDER_MODES)
                elif key == "\\":
                    show_debug = not show_debug
                elif key == "q" or key == "\x1b":
                    if cam:
                        cam.close()
                    return configs      # quit wizard early

                # Throttle to ~60 FPS
                _el = time.monotonic() - _t_now
                if _el < 0.016:
                    time.sleep(0.016 - _el)

            elif phase == "name":
                # Name-input phase: hand off to _prompt_line for text editing.
                _draw_text(buf, prompt_row + 1, 2,
                           "\x1b[33m[Enter]\x1b[0m accept  "
                           "\x1b[33m[Esc]\x1b[0m back to preview")

                out.write(
                    _SY_S + b"\x1b[H\x1b[2J" + buf.getvalue() + _SY_E)
                out.flush()

                result = _prompt_line(
                    term, out, prompt_row, 2,
                    "Channel name: ", default_name)

                if result is None:
                    phase = "preview"   # back to live preview
                    _needs_clear = True
                else:
                    chosen_name = result

        if cam:
            cam.close()

        if chosen_name:
            configs.append(CameraConfig(
                name=chosen_name,
                type=dev.dtype,
                device=dev.device_id,
                width=dev.properties.get("width", 640),
                height=dev.properties.get("height", 480),
                fps=dev.properties.get("fps", 30),
            ))

    return configs


def _screen_robots(term: _Term, out, devices: list[DetectedDevice]
                   ) -> list[RobotConfig]:
    """Robot identification screen — oscillation + name prompt."""
    configs: list[RobotConfig] = []
    show_debug = False
    _t_prev = time.monotonic()
    _fps = 0.0

    for i, dev in enumerate(devices):
        rob = _make_robot(dev)
        chosen_name: str | None = None
        chosen_role: str = "follower"
        default_name = f"arm_{i}"
        phase = "preview"    # "preview" → "name" → "role" → done
        _needs_clear = True

        while chosen_name is None or phase != "done":
            W, H = term.cols, term.rows

            buf = io.BytesIO()

            _draw_header(buf, W, 2, 3, "Robots")

            _draw_text(buf, 3, 2,
                       f"Robot {i+1}/{len(devices)}: "
                       f"\x1b[96m{dev.label}\x1b[0m")
            _draw_text(buf, 4, 2,
                       f"Type: {dev.dtype}  Joints: "
                       f"{dev.properties.get('num_joints', '?')}")

            if dev.dtype == "pseudo":
                _draw_text(buf, 5, 2,
                           "\x1b[93m⟳ Last joint oscillating "
                           "(simulated identification)\x1b[0m")

            # Robot state display (re-read every iteration)
            if rob is not None:
                _, state = rob.read()
                pos = state.get("position", np.zeros(0))
                start_row = 7
                n_joints = len(pos)
                bar_w = min(40, W - 25)
                for j in range(n_joints):
                    p = pos[j]
                    frac = float(np.clip((p + 2) / 4, 0, 1))
                    bar_len = int(frac * bar_w)
                    bar = "█" * bar_len + "░" * (bar_w - bar_len)
                    marker = " \x1b[91m← moving\x1b[0m" if j == n_joints - 1 else ""
                    _draw_text(buf, start_row + j, 4,
                               f"j{j} \x1b[36m{p:+6.2f}\x1b[0m "
                               f"\x1b[33m{bar}\x1b[0m{marker}")

            prompt_row = max(7 + dev.properties.get("num_joints", 6) + 2,
                             H - 5)

            if phase == "preview":
                # Live-preview phase: robot state refreshes every iteration,
                # shortcut keys are handled directly.
                _draw_text(buf, prompt_row, 2,
                           "\x1b[33m[Enter]\x1b[0m name  "
                           "\x1b[33m[s]\x1b[0m skip  "
                           "\x1b[33m[\\]\x1b[0m debug  "
                           "\x1b[33m[Esc/q]\x1b[0m quit")

                # FPS tracking
                _t_now = time.monotonic()
                _dt = _t_now - _t_prev
                _t_prev = _t_now
                if _dt > 0:
                    _fps = 0.9 * _fps + 0.1 / _dt

                if show_debug:
                    _draw_text(buf, 2, W - 18,
                               f"\x1b[48;5;234m\x1b[38;5;82m FPS: {_fps:5.1f} \x1b[0m")

                buf.write(f"\x1b[{prompt_row + 1};1H\x1b[J".encode())

                _clear = b"\x1b[2J" if _needs_clear else b""
                _needs_clear = False
                out.write(
                    _SY_S + b"\x1b[H" + _clear + buf.getvalue() + _SY_E)
                out.flush()

                key = term.read_key()
                if key == "\n" or key == "\r":
                    phase = "name"
                    _needs_clear = True
                elif key == "s":
                    break               # skip this robot
                elif key == "\\":
                    show_debug = not show_debug
                elif key == "q" or key == "\x1b":
                    if rob:
                        rob.close()
                    return configs

                # Throttle to ~60 FPS
                _el = time.monotonic() - _t_now
                if _el < 0.016:
                    time.sleep(0.016 - _el)

            elif phase == "name":
                _draw_text(buf, prompt_row + 1, 2,
                           "\x1b[33m[Enter]\x1b[0m accept  "
                           "\x1b[33m[Esc]\x1b[0m back")

                out.write(
                    _SY_S + b"\x1b[H\x1b[2J" + buf.getvalue() + _SY_E)
                out.flush()

                result = _prompt_line(
                    term, out, prompt_row, 2,
                    "Channel name: ", default_name)
                if result is None:
                    phase = "preview"
                    _needs_clear = True
                    continue
                chosen_name = result
                phase = "role"

            elif phase == "role":
                _draw_text(buf, prompt_row, 2,
                           f"Name: \x1b[97;1m{chosen_name}\x1b[0m")
                _draw_text(buf, prompt_row + 1, 2,
                           "\x1b[33m[Enter]\x1b[0m accept  "
                           "\x1b[33m[Esc]\x1b[0m back")

                out.write(
                    _SY_S + b"\x1b[H\x1b[2J" + buf.getvalue() + _SY_E)
                out.flush()

                result = _prompt_line(
                    term, out, prompt_row - 1, 2,
                    "Role ([f]ollower / [l]eader): ", "f")
                if result is None:
                    phase = "name"
                    continue
                chosen_role = "leader" if result.lower().startswith("l") else "follower"
                phase = "done"

        if rob:
            rob.close()

        if chosen_name:
            configs.append(RobotConfig(
                name=chosen_name,
                type=dev.dtype,
                role=chosen_role,
                num_joints=dev.properties.get("num_joints", 6),
            ))

    return configs


def _screen_project(term: _Term, out) -> tuple[str, str] | None:
    """Project settings screen — name + storage root."""
    W, H = term.cols, term.rows

    out.write(_SY_S + b"\x1b[2J")
    _draw_header(out, W, 3, 3, "Project Settings")

    _draw_text(out, 4, 2,
               "Configure project name and storage location.")
    out.write(_SY_E)
    out.flush()

    name = _prompt_line(term, out, 6, 2, "Project name: ", "default")
    if name is None:
        return None

    storage = _prompt_line(term, out, 8, 2, "Storage root: ", "~/rollio_data")
    if storage is None:
        return None

    return name, storage


def _screen_summary(term: _Term, out,
                    cam_configs: list[CameraConfig],
                    rob_configs: list[RobotConfig],
                    cam_devs: list[DetectedDevice],
                    rob_devs: list[DetectedDevice],
                    project_name: str,
                    storage_root: str,
                    output_path: str) -> bool:
    """Live summary screen with camera previews + robot states.

    Returns True to save, False to cancel.
    """
    # Build sensors from configs - match by device_id first, then by type
    # Track which devices have been used to avoid double-matching
    cam_sensors = []
    used_cam_devs: set[int] = set()
    for cc in cam_configs:
        sensor = None
        # First try exact device_id match
        for di, d in enumerate(cam_devs):
            if di not in used_cam_devs and str(d.device_id) == str(cc.device):
                sensor = _make_camera(d)
                used_cam_devs.add(di)
                break
        # Fallback: match by type if no device match
        if sensor is None:
            for di, d in enumerate(cam_devs):
                if di not in used_cam_devs and d.dtype == cc.type:
                    sensor = _make_camera(d)
                    used_cam_devs.add(di)
                    break
        cam_sensors.append((cc, sensor))

    rob_sensors = []
    used_rob_devs: set[int] = set()
    for rc in rob_configs:
        sensor = None
        for di, d in enumerate(rob_devs):
            if di not in used_rob_devs and d.dtype == rc.type:
                sensor = _make_robot(d)
                used_rob_devs.add(di)
                break
        rob_sensors.append((rc, sensor))

    mode_idx = 2  # "gray" mode for compact previews
    show_debug = False
    _t_prev = time.monotonic()
    _fps = 0.0
    result = None
    _needs_clear = True

    def _draw_text_clear(buf, row: int, col: int, text: str, clear_w: int = 0):
        """Draw text and clear to specified width or end of line."""
        if clear_w > 0:
            # Calculate visible length (strip ANSI codes)
            import re
            vis_len = len(re.sub(r"\x1b\[[0-9;]*m", "", text))
            padding = " " * max(0, clear_w - vis_len)
            buf.write(f"\x1b[{row};{col}H{text}{padding}\x1b[0m".encode())
        else:
            buf.write(f"\x1b[{row};{col}H{text}\x1b[K\x1b[0m".encode())

    try:
        while result is None:
            W, H = term.cols, term.rows
            buf = io.BytesIO()

            # FPS tracking
            _t_now = time.monotonic()
            _dt = _t_now - _t_prev
            _t_prev = _t_now
            if _dt > 0:
                _fps = 0.9 * _fps + 0.1 / _dt

            _draw_header(buf, W, 3, 3, "Summary — Live Preview")

            # ── Layout: left panel (info), right panel (live previews) ──
            info_w = max(40, W // 3)
            preview_w = W - info_w - 2

            # ── Left panel: configuration info ──
            row = 3
            box_line = "─" * (info_w - 4)
            _draw_text_clear(buf, row, 2, "┌" + box_line + "┐", info_w)
            row += 1
            _draw_text_clear(buf, row, 2,
                             f"│ \x1b[1mProject:\x1b[0m \x1b[97;1m{project_name[:info_w-14]}\x1b[0m",
                             info_w)
            row += 1
            _draw_text_clear(buf, row, 2,
                             f"│ \x1b[1mStorage:\x1b[0m \x1b[90m{storage_root[:info_w-14]}\x1b[0m",
                             info_w)
            row += 1
            _draw_text_clear(buf, row, 2, "├" + box_line + "┤", info_w)
            row += 1
            _draw_text_clear(buf, row, 2,
                             f"│ \x1b[1;96mCameras ({len(cam_configs)})\x1b[0m",
                             info_w)
            row += 1
            for ci, (cc, _) in enumerate(cam_sensors):
                _draw_text_clear(buf, row, 2,
                                 f"│  \x1b[96m{cc.name[:12]:<12}\x1b[0m "
                                 f"\x1b[90m{cc.type} {cc.width}×{cc.height}\x1b[0m",
                                 info_w)
                row += 1
            _draw_text_clear(buf, row, 2, "├" + box_line + "┤", info_w)
            row += 1
            _draw_text_clear(buf, row, 2,
                             f"│ \x1b[1;93mRobots ({len(rob_configs)})\x1b[0m",
                             info_w)
            row += 1
            for ri, (rc, _) in enumerate(rob_sensors):
                role_clr = "92" if rc.role == "leader" else "33"
                _draw_text_clear(buf, row, 2,
                                 f"│  \x1b[93m{rc.name[:12]:<12}\x1b[0m "
                                 f"\x1b[{role_clr}m{rc.role}\x1b[0m "
                                 f"\x1b[90m{rc.num_joints}-DOF\x1b[0m",
                                 info_w)
                row += 1
            _draw_text_clear(buf, row, 2, "├" + box_line + "┤", info_w)
            row += 1
            _draw_text_clear(buf, row, 2,
                             f"│ \x1b[1mSave to:\x1b[0m", info_w)
            row += 1
            path_display = output_path[:info_w - 6]
            _draw_text_clear(buf, row, 2,
                             f"│  \x1b[90m{path_display}\x1b[0m", info_w)
            row += 1
            _draw_text_clear(buf, row, 2, "└" + box_line + "┘", info_w)

            # ── Right panel: live camera previews ──
            preview_col = info_w + 2
            preview_row = 3
            n_cams = len(cam_sensors)
            n_robs = len(rob_sensors)

            # Calculate space for cameras and robots
            avail_h = max(4, H - 6)
            cam_h_total = max(4, (avail_h * 2) // 3) if n_cams else 0
            rob_h_total = avail_h - cam_h_total if n_robs else 0

            # Draw camera previews
            if n_cams > 0:
                # Reserve 1 row for label below each preview
                cam_h_each = max(4, cam_h_total // n_cams)
                mode = RENDER_MODES[mode_idx]
                _draw_text(buf, preview_row, preview_col,
                           f"\x1b[1;96m{'─── CAMERAS ':─<{preview_w}}\x1b[0m")
                preview_row += 1

                for ci, (cc, sensor) in enumerate(cam_sensors):
                    preview_h = cam_h_each - 2  # leave room for label
                    if sensor is not None:
                        try:
                            _, frame = sensor.read()
                            if frame is not None:
                                # Calculate size preserving aspect
                                fh, fw = frame.shape[:2]
                                rw, rh = calc_render_size(
                                    fw, fh, preview_w - 2, preview_h)
                                rendered = render_frame(frame, rw, rh, mode)
                                buf.write(blit_frame(
                                    rendered, preview_row, preview_col + 1))
                        except Exception:
                            _draw_text(buf, preview_row, preview_col + 1,
                                       f"\x1b[90m(error reading frame)\x1b[0m")
                    else:
                        _draw_text(buf, preview_row, preview_col + 1,
                                   f"\x1b[90m(no preview available)\x1b[0m")

                    # Draw camera label below preview
                    label_row = preview_row + preview_h
                    _draw_text_clear(buf, label_row, preview_col + 1,
                                     f"\x1b[96;1m{cc.name}\x1b[0m "
                                     f"\x1b[90m({cc.type} {cc.width}×{cc.height})\x1b[0m",
                                     preview_w - 2)
                    preview_row += cam_h_each

            # Draw robot states
            if n_robs > 0:
                _draw_text(buf, preview_row, preview_col,
                           f"\x1b[1;93m{'─── ROBOTS ':─<{preview_w}}\x1b[0m")
                preview_row += 1
                bar_w = min(30, preview_w - 20)

                for ri, (rc, sensor) in enumerate(rob_sensors):
                    role_clr = "92" if rc.role == "leader" else "33"
                    _draw_text(buf, preview_row, preview_col,
                               f"\x1b[{role_clr};1m{rc.name}\x1b[0m "
                               f"\x1b[90m({rc.num_joints}-DOF, {rc.role})\x1b[0m")

                    if sensor is not None:
                        try:
                            _, state = sensor.read()
                            pos = state.get("position", np.zeros(0))
                            # Show all joints
                            for j in range(len(pos)):
                                p = pos[j]
                                frac = float(np.clip((p + 2) / 4, 0, 1))
                                bar_len = int(frac * bar_w)
                                bar = "█" * bar_len + "░" * (bar_w - bar_len)
                                _draw_text(buf, preview_row + 1 + j,
                                           preview_col + 2,
                                           f"j{j} \x1b[36m{p:+5.2f}\x1b[0m "
                                           f"\x1b[33m{bar}\x1b[0m")
                            preview_row += 1 + len(pos) + 1  # header + joints + spacing
                        except Exception:
                            _draw_text(buf, preview_row + 1, preview_col + 2,
                                       "\x1b[90m(error)\x1b[0m")
                            preview_row += 3
                    else:
                        _draw_text(buf, preview_row + 1, preview_col + 2,
                                   "\x1b[90m(no preview)\x1b[0m")
                        preview_row += 3

            # Debug overlay
            if show_debug:
                _draw_text(buf, 2, W - 18,
                           f"\x1b[48;5;234m\x1b[38;5;82m FPS: {_fps:5.1f} \x1b[0m")

            # ── Footer / controls ──
            footer_row = H - 2
            _draw_text_clear(buf, footer_row, 2,
                             "\x1b[33m[Enter]\x1b[0m save config  "
                             "\x1b[33m[m]\x1b[0m preview mode  "
                             "\x1b[33m[\\]\x1b[0m debug  "
                             "\x1b[33m[q/Esc]\x1b[0m cancel", 0)

            # Erase below footer
            buf.write(f"\x1b[{footer_row + 1};1H\x1b[J".encode())

            # Full clear on first frame, then incremental
            _clear = b"\x1b[2J" if _needs_clear else b""
            _needs_clear = False
            out.write(_SY_S + _clear + b"\x1b[H" + buf.getvalue() + _SY_E)
            out.flush()

            # Handle input (non-blocking)
            key = term.read_key()
            if key == "\n" or key == "\r":
                result = True
            elif key == "q" or key == "\x1b":
                result = False
            elif key == "m":
                mode_idx = (mode_idx + 1) % len(RENDER_MODES)
            elif key == "\\":
                show_debug = not show_debug

            # Throttle to ~30 FPS
            _el = time.monotonic() - _t_now
            if _el < 0.033:
                time.sleep(0.033 - _el)

    finally:
        # Clean up sensors
        for _, sensor in cam_sensors:
            if sensor is not None:
                try:
                    sensor.close()
                except Exception:
                    pass
        for _, sensor in rob_sensors:
            if sensor is not None:
                try:
                    sensor.close()
                except Exception:
                    pass

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Main wizard entry point
# ═══════════════════════════════════════════════════════════════════════

def run_wizard(output_path: str) -> RollioConfig | None:
    """Run the interactive setup wizard.  Returns config or None on abort."""
    print("Scanning for hardware…")
    cam_devs = scan_cameras()
    rob_devs = scan_robots()
    print(f"  Found {len(cam_devs)} camera(s), {len(rob_devs)} robot(s).")
    print("Launching wizard TUI…")
    time.sleep(0.5)

    out = sys.stdout.buffer

    with _Term() as term:
        # Step 1: Cameras
        cam_configs = _screen_cameras(term, out, cam_devs)
        if not cam_configs:
            cam_configs = [CameraConfig()]  # fallback pseudo

        # Step 2: Robots
        rob_configs = _screen_robots(term, out, rob_devs)
        if not rob_configs:
            rob_configs = [RobotConfig()]  # fallback pseudo

        # Step 3: Project settings
        proj = _screen_project(term, out)
        if proj is None:
            return None
        project_name, storage_root = proj

        # Step 4: Live summary screen with previews
        should_save = _screen_summary(
            term, out,
            cam_configs, rob_configs,
            cam_devs, rob_devs,
            project_name, storage_root,
            output_path)

        if not should_save:
            return None

    return RollioConfig(
        project_name=project_name,
        cameras=cam_configs,
        robots=rob_configs,
        storage=StorageConfig(root=storage_root),
    )
