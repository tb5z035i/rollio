"""Interactive TUI Setup Wizard.

Scans hardware, shows live camera preview / robot oscillation, and
prompts the user for channel names.  Works without a desktop environment.
"""
from __future__ import annotations

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
from rollio.tui.renderer import render_frame

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
    W, H = term.cols, term.rows

    for i, dev in enumerate(devices):
        cam = _make_camera(dev)
        chosen_name: str | None = None

        default_name = f"cam_{i}"
        name_buf = default_name

        while chosen_name is None:
            t0 = time.monotonic()
            W, H = term.cols, term.rows

            frame_out = bytearray()
            frame_out.extend(_SY_S)
            frame_out.extend(b"\x1b[2J")

            # Header
            _draw_header(out, W, 1, total_steps, "Cameras")

            # Device info
            _draw_text(out, 3, 2,
                       f"Camera {i+1}/{len(devices)}: "
                       f"\x1b[96m{dev.label}\x1b[0m")
            _draw_text(out, 4, 2,
                       f"Type: {dev.dtype}  Device: {dev.device_id}")

            # Live preview
            preview_y = 6
            preview_h = max(4, H - 14)
            preview_w = min(W - 4, 120)
            if cam is not None:
                _, frame = cam.read()
                if frame is not None:
                    rendered = render_frame(frame, preview_w, preview_h)
                    # Position the preview
                    lines = rendered.split(b"\x1b[0m\n")
                    for li, line in enumerate(lines):
                        out.write(f"\x1b[{preview_y + li};3H".encode())
                        out.write(line)
                        out.write(b"\x1b[0m")
            else:
                _draw_text(out, preview_y, 3, "(no preview available)")

            # Prompt area
            prompt_row = preview_y + preview_h + 1
            _draw_text(out, prompt_row + 1, 2,
                       "\x1b[33m[Enter]\x1b[0m accept  "
                       "\x1b[33m[s]\x1b[0m skip  "
                       "\x1b[33m[Esc/q]\x1b[0m quit wizard")

            out.write(_SY_E)
            out.flush()

            # Name input
            result = _prompt_line(
                term, out, prompt_row, 2,
                "Channel name: ", name_buf)

            if result is None:
                # Quit or skip
                key = term.read_key_blocking(0.01)
                if key == "s":
                    break   # skip this camera
                if cam:
                    cam.close()
                return configs  # quit wizard early
            else:
                chosen_name = result
                name_buf = result

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
    W, H = term.cols, term.rows

    for i, dev in enumerate(devices):
        rob = _make_robot(dev)
        chosen_name: str | None = None
        chosen_role: str = "follower"
        default_name = f"arm_{i}"
        phase = "name"       # "name" → "role" → done

        while chosen_name is None or phase != "done":
            t0 = time.monotonic()
            W, H = term.cols, term.rows

            out.write(_SY_S)
            out.write(b"\x1b[2J")

            _draw_header(out, W, 2, 3, "Robots")

            _draw_text(out, 3, 2,
                       f"Robot {i+1}/{len(devices)}: "
                       f"\x1b[96m{dev.label}\x1b[0m")
            _draw_text(out, 4, 2,
                       f"Type: {dev.dtype}  Joints: "
                       f"{dev.properties.get('num_joints', '?')}")

            if dev.dtype == "pseudo":
                _draw_text(out, 5, 2,
                           "\x1b[93m⟳ Last joint oscillating "
                           "(simulated identification)\x1b[0m")

            # Robot state display
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
                    _draw_text(out, start_row + j, 4,
                               f"j{j} \x1b[36m{p:+6.2f}\x1b[0m "
                               f"\x1b[33m{bar}\x1b[0m{marker}")

            prompt_row = max(7 + dev.properties.get("num_joints", 6) + 2,
                             H - 5)

            _draw_text(out, prompt_row + 1, 2,
                       "\x1b[33m[Enter]\x1b[0m accept  "
                       "\x1b[33m[s]\x1b[0m skip  "
                       "\x1b[33m[Esc/q]\x1b[0m quit")

            out.write(_SY_E)
            out.flush()

            if phase == "name":
                result = _prompt_line(
                    term, out, prompt_row, 2,
                    "Channel name: ", default_name)
                if result is None:
                    if rob:
                        rob.close()
                    return configs
                chosen_name = result
                phase = "role"
            elif phase == "role":
                _draw_text(out, prompt_row, 2,
                           f"Name: \x1b[97;1m{chosen_name}\x1b[0m")
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

        # Summary screen
        out.write(b"\x1b[2J")
        _draw_header(out, term.cols, 3, 3, "Summary")
        _draw_text(out, 3, 2, f"Project: \x1b[97;1m{project_name}\x1b[0m")
        _draw_text(out, 4, 2, f"Storage: \x1b[97;1m{storage_root}\x1b[0m")
        _draw_text(out, 6, 2, "Cameras:")
        for ci, c in enumerate(cam_configs):
            _draw_text(out, 7 + ci, 4,
                       f"\x1b[96m{c.name}\x1b[0m ({c.type} "
                       f"{c.width}×{c.height}@{c.fps}fps)")
        rob_start = 8 + len(cam_configs)
        _draw_text(out, rob_start, 2, "Robots:")
        for ri, r in enumerate(rob_configs):
            _draw_text(out, rob_start + 1 + ri, 4,
                       f"\x1b[96m{r.name}\x1b[0m ({r.type} "
                       f"{r.num_joints}-DOF, {r.role})")

        save_row = rob_start + 2 + len(rob_configs)
        _draw_text(out, save_row, 2,
                   f"Config will be saved to: \x1b[97;1m{output_path}\x1b[0m")
        _draw_text(out, save_row + 2, 2,
                   "\x1b[33m[Enter]\x1b[0m save  "
                   "\x1b[33m[q]\x1b[0m cancel")
        out.flush()

        # Wait for confirmation
        while True:
            key = term.read_key_blocking(0.1)
            if key == "\n" or key == "\r":
                break
            if key == "q" or key == "\x1b":
                return None

    return RollioConfig(
        project_name=project_name,
        cameras=cam_configs,
        robots=rob_configs,
        storage=StorageConfig(root=storage_root),
    )
