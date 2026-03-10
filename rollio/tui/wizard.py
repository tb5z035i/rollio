"""Interactive TUI Setup Wizard.

Scans hardware, shows live camera preview / robot oscillation, and
prompts the user for channel names.  Works without a desktop environment.
"""
from __future__ import annotations

import io
import os
import select
import signal
import sys
import termios
import time
import tty

import cv2
import numpy as np

from rollio.collect import AsyncCollectionRuntime
from rollio.config.pairing import (
    default_mapper_for_pair,
    suggest_teleop_pairs,
    validate_teleop_pairs,
)
from rollio.config.schema import (
    CameraConfig, ControlConfig, RobotConfig, RollioConfig, StorageConfig,
    TeleopPairConfig, EncoderConfig,
)
from rollio.episode.codecs import (
    available_depth_codec_options,
    available_rgb_codec_options,
)
from rollio.sensors.base import CameraFormat, CameraMode, ImageSensor
from rollio.sensors.pseudo_camera import PseudoCamera
from rollio.sensors.pseudo_robot import PseudoRobot
from rollio.sensors.realsense_camera import RealSenseCamera
from rollio.sensors.scanner import DetectedDevice, scan_cameras, scan_robots
from rollio.sensors.v4l2_camera import V4L2Camera
from rollio.tui.renderer import (
    RENDER_MODES, DEPTH_MODES, MODE_LABELS,
    blit_frame, calc_render_size, render_frame, render_depth,
)

MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("teleop", "Tele-operation"),
    ("intervention", "Intervention"),
)

MAPPER_OPTIONS: tuple[tuple[str, str], ...] = (
    ("joint_direct", "Direct joint mapping"),
    ("pose_fk_ik", "FK-IK pose mapping"),
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
        try:
            cols, rows = os.get_terminal_size()
        except OSError:
            cols, rows = 80, 24
        self.cols = max(cols, 40)
        self.rows = max(rows, 10)

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

def _make_camera(dev: DetectedDevice,
                 width: int | None = None,
                 height: int | None = None,
                 fps: int | None = None,
                 pixel_format: str | None = None) -> ImageSensor | None:
    """Instantiate a camera sensor from a DetectedDevice for preview.

    If width/height/fps/pixel_format are provided, use them; otherwise
    use device defaults.
    """
    w = width or dev.properties.get("width", 640)
    h = height or dev.properties.get("height", 480)
    f = fps or dev.properties.get("fps", 30)

    if dev.dtype == "pseudo":
        cam = PseudoCamera(name="preview", width=w, height=h, fps=f)
        cam.open()
        return cam
    elif dev.dtype == "v4l2":
        pf = pixel_format or "MJPG"
        try:
            cam = V4L2Camera(
                name="preview",
                device=dev.device_id,
                width=w, height=h, fps=f,
                pixel_format=pf)
            cam.open()
            return cam
        except Exception:
            pass
    elif dev.dtype.startswith("realsense_"):
        # Extract channel and serial from device_id (format: "serial:channel")
        channel = dev.properties.get("channel", "color")
        serial = dev.properties.get("serial", str(dev.device_id).split(":")[0])
        pf = pixel_format or dev.pixel_format
        try:
            # Build kwargs based on which channel is enabled
            kwargs = {
                "name": "preview",
                "device": serial,
                "enable_color": (channel == "color"),
                "enable_depth": (channel == "depth"),
                "enable_infrared": (channel == "infrared"),
                "preview_channel": channel,
            }
            # Set the resolution and format for the correct channel
            if channel == "color":
                kwargs.update(width=w, height=h, fps=f)
            elif channel == "depth":
                kwargs.update(depth_width=w, depth_height=h, depth_fps=f)
                if pf:
                    kwargs["depth_format"] = pf
            elif channel == "infrared":
                kwargs.update(ir_width=w, ir_height=h, ir_fps=f)
                if pf:
                    kwargs["ir_format"] = pf

            cam = RealSenseCamera(**kwargs)
            cam.open()
            return cam
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


def _wait_for_keypress(term: _Term) -> str:
    """Block until a single key is pressed."""
    while True:
        key = term.read_key_blocking(0.05)
        if key is not None:
            return key


def _teleop_warning_lines(robots: list[RobotConfig]) -> list[str] | None:
    """Return warning text when tele-operation prerequisites are not met."""
    leaders = [robot for robot in robots if robot.role == "leader"]
    followers = [robot for robot in robots if robot.role == "follower"]
    if leaders and followers:
        return None
    if not robots:
        return [
            "\x1b[91mTele-operation requires at least one leader and one follower robot.\x1b[0m",
            "No robots were configured in the previous step.",
        ]
    return [
        "\x1b[91mTele-operation requires at least one leader and one follower robot.\x1b[0m",
        f"Configured robots: {len(leaders)} leader(s), {len(followers)} follower(s).",
    ]


def _show_warning_screen(
    term: _Term,
    out,
    *,
    title: str,
    lines: list[str],
    prompt: str,
    step: int,
    total_steps: int,
) -> None:
    """Render a blocking warning page."""
    W = term.cols
    out.write(_SY_S + b"\x1b[2J")
    _draw_header(out, W, step, total_steps, title)
    row = 4
    for line in lines:
        _draw_text(out, row, 2, line)
        row += 1
    _draw_text(out, row + 1, 2, prompt)
    out.write(_SY_E)
    out.flush()
    _wait_for_keypress(term)


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


def _pick_format(term: _Term, out, formats: list[CameraFormat],
                 current_idx: int, W: int, H: int) -> int | None:
    """Display format picker.

    Returns selected index, or None if cancelled (Esc).
    Uses same UX as resolution picker (enter number + press Enter).
    """
    if not formats:
        return None

    input_buf = ""

    while True:
        buf = io.BytesIO()
        buf.write(b"\x1b[2J\x1b[H")

        # Header
        buf.write(f"\x1b[1;1H\x1b[48;5;24m\x1b[97;1m{'  SELECT FORMAT  ':<{W}}\x1b[0m".encode())

        # Instructions
        _draw_text(buf, 3, 2,
                   f"Enter number and press Enter, or \x1b[33m[Esc]\x1b[0m to cancel")
        _draw_text(buf, 4, 2,
                   f"Input: \x1b[1;97m{input_buf}\x1b[5m_\x1b[0m")

        # List formats
        start_row = 6
        for idx, fmt in enumerate(formats):
            is_cur = (idx == current_idx)
            num_str = f"{idx + 1:>2}"
            n_modes = len(fmt.modes)
            if is_cur:
                text = (f"\x1b[1;92m[{num_str}] {fmt.fourcc}\x1b[0m "
                        f"\x1b[90m({fmt.description}, {n_modes} modes)\x1b[0m")
            else:
                text = (f"\x1b[33m[{num_str}]\x1b[0m {fmt.fourcc} "
                        f"\x1b[90m({fmt.description}, {n_modes} modes)\x1b[0m")
            _draw_text(buf, start_row + idx, 4, text)

        out.write(_SY_S + buf.getvalue() + _SY_E)
        out.flush()

        key = term.read_key_blocking(0.05)
        if key is None:
            continue
        elif key == "\x1b":  # Escape
            return None
        elif key == "\n" or key == "\r":
            # Try to parse input
            if input_buf:
                try:
                    sel = int(input_buf) - 1
                    if 0 <= sel < len(formats):
                        return sel
                except ValueError:
                    pass
            input_buf = ""  # Clear on invalid
        elif key == "\x7f" or key == "\x08":  # Backspace
            input_buf = input_buf[:-1]
        elif key.isdigit():
            input_buf += key


def _pick_resolution(term: _Term, out, modes: list[CameraMode],
                     current_idx: int, W: int, H: int) -> int | None:
    """Display resolution picker grid.

    Each item is a separate W×H@FPS tuplet (standard v4l2 form).
    Returns selected index, or None if cancelled (Esc).
    """
    # Sort modes: by resolution (largest first), then by FPS (highest first)
    sorted_modes = sorted(modes,
                          key=lambda m: (m.width * m.height, m.fps),
                          reverse=True)

    # Build display items: (idx_in_original_modes, display_str, is_current)
    items: list[tuple[int, str, bool]] = []
    for m in sorted_modes:
        orig_idx = modes.index(m)
        is_cur = (orig_idx == current_idx)
        display = f"{m.width}×{m.height}@{m.fps}"
        items.append((orig_idx, display, is_cur))

    # Calculate grid layout
    max_item_w = max(len(it[1]) for it in items) + 6  # "[NN] " + padding
    cols = max(1, (W - 4) // max_item_w)
    rows = (len(items) + cols - 1) // cols

    input_buf = ""
    scroll_offset = 0
    visible_rows = max(4, H - 8)

    while True:
        buf = io.BytesIO()
        buf.write(b"\x1b[2J\x1b[H")

        # Header
        buf.write(f"\x1b[1;1H\x1b[48;5;24m\x1b[97;1m{'  SELECT RESOLUTION  ':<{W}}\x1b[0m".encode())

        # Instructions
        _draw_text(buf, 3, 2,
                   f"Enter number and press Enter, or \x1b[33m[Esc]\x1b[0m to cancel")
        _draw_text(buf, 4, 2,
                   f"Input: \x1b[1;97m{input_buf}\x1b[5m_\x1b[0m")

        # Grid
        start_row = 6
        for row_i in range(min(visible_rows, rows - scroll_offset)):
            actual_row = row_i + scroll_offset
            y = start_row + row_i
            for col_i in range(cols):
                idx = actual_row * cols + col_i
                if idx >= len(items):
                    break
                mode_idx_val, display, is_cur = items[idx]
                x = 2 + col_i * max_item_w

                num_str = f"{idx + 1:>2}"
                if is_cur:
                    # Highlight current selection
                    text = f"\x1b[1;92m[{num_str}] {display}\x1b[0m"
                else:
                    text = f"\x1b[33m[{num_str}]\x1b[0m {display}"
                _draw_text(buf, y, x, text)

        # Scroll indicator
        if rows > visible_rows:
            _draw_text(buf, start_row + visible_rows, 2,
                       f"\x1b[90m({scroll_offset + 1}-{min(scroll_offset + visible_rows, rows)}"
                       f" of {rows} rows, ↑/↓ to scroll)\x1b[0m")

        out.write(_SY_S + buf.getvalue() + _SY_E)
        out.flush()

        key = term.read_key_blocking(0.05)
        if key is None:
            continue
        elif key == "\x1b":  # Escape
            return None
        elif key == "\n" or key == "\r":
            # Try to parse input
            if input_buf:
                try:
                    sel = int(input_buf) - 1
                    if 0 <= sel < len(items):
                        return items[sel][0]  # Return mode index
                except ValueError:
                    pass
            input_buf = ""  # Clear on invalid
        elif key == "\x7f" or key == "\x08":  # Backspace
            input_buf = input_buf[:-1]
        elif key.isdigit():
            input_buf += key
        elif key == "[" and scroll_offset > 0:  # Scroll up ([ key)
            scroll_offset = max(0, scroll_offset - 1)
        elif key == "]" and scroll_offset < rows - visible_rows:  # Scroll down (] key)
            scroll_offset = min(rows - visible_rows, scroll_offset + 1)


def _pick_option(
    term: _Term,
    out,
    *,
    title: str,
    options: list[str],
    current_idx: int = 0,
    subtitle: str = "",
) -> int | None:
    """Display a simple numbered option picker."""
    if not options:
        return None

    input_buf = ""
    while True:
        W, H = term.cols, term.rows
        buf = io.BytesIO()
        buf.write(b"\x1b[2J\x1b[H")
        buf.write(f"\x1b[1;1H\x1b[48;5;24m\x1b[97;1m{f'  {title}  ':{W}}\x1b[0m".encode())
        if subtitle:
            _draw_text(buf, 3, 2, subtitle)
        _draw_text(
            buf,
            5,
            2,
            "Enter number and press Enter, or "
            "\x1b[33m[Esc]\x1b[0m to cancel",
        )
        _draw_text(buf, 6, 2, f"Input: \x1b[1;97m{input_buf}\x1b[5m_\x1b[0m")

        start_row = 8
        for idx, option in enumerate(options):
            marker = "\x1b[1;92m" if idx == current_idx else "\x1b[33m"
            text = f"{marker}[{idx + 1:>2}]\x1b[0m {option}"
            _draw_text(buf, start_row + idx, 4, text)

        out.write(_SY_S + buf.getvalue() + _SY_E)
        out.flush()

        key = term.read_key_blocking(0.05)
        if key is None:
            continue
        if key == "\x1b":
            return None
        if key == "\n" or key == "\r":
            if not input_buf:
                return current_idx
            try:
                selected = int(input_buf) - 1
            except ValueError:
                input_buf = ""
                continue
            if 0 <= selected < len(options):
                return selected
            input_buf = ""
            continue
        if key == "\x7f" or key == "\x08":
            input_buf = input_buf[:-1]
        elif key.isdigit():
            input_buf += key


def _screen_cameras(term: _Term, out, devices: list[DetectedDevice]
                    , total_steps: int = 5) -> list[CameraConfig]:
    """Camera identification screen — live preview + name prompt."""
    configs: list[CameraConfig] = []

    mode_idx = 0   # start at "true" (24-bit truecolor)
    show_debug = False
    _t_prev = time.monotonic()
    _fps = 0.0

    for i, dev in enumerate(devices):
        # Get available formats for this device
        formats = dev.formats if dev.formats else []
        format_idx = 0
        mode_sel_idx = 0  # selected mode within current format

        # Current camera settings (use DetectedDevice fields directly)
        cur_width = dev.width
        cur_height = dev.height
        cur_fps = dev.fps
        cur_format = dev.pixel_format

        # Depth/IR visualization mode (for realsense_depth, realsense_infrared)
        depth_mode_idx = 0  # 0 = turbo colormap, 1 = gray
        is_depth_camera = dev.dtype in ("realsense_depth", "realsense_infrared")

        # Try to find MJPG format as default (usually better for USB cameras)
        for fi, fmt in enumerate(formats):
            if fmt.fourcc == "MJPG":
                format_idx = fi
                if fmt.modes:
                    # Find a reasonable default mode (640x480 or closest)
                    for mi, m in enumerate(fmt.modes):
                        if m.width == 640 and m.height == 480:
                            mode_sel_idx = mi
                            cur_width, cur_height, cur_fps = m.width, m.height, m.fps
                            break
                    else:
                        mode_sel_idx = 0
                        m = fmt.modes[0]
                        cur_width, cur_height, cur_fps = m.width, m.height, m.fps
                cur_format = fmt.fourcc
                break

        cam = _make_camera(dev, cur_width, cur_height, cur_fps, cur_format)
        chosen_name: str | None = None
        default_name = f"cam_{i}"
        phase = "preview"          # "preview" → "name" → accepted
        _needs_clear = True
        _needs_reopen = False

        while chosen_name is None:
            # Reopen camera if settings changed
            if _needs_reopen:
                if cam:
                    cam.close()
                cam = _make_camera(dev, cur_width, cur_height, cur_fps, cur_format)
                _needs_reopen = False
                _needs_clear = True

            W, H = term.cols, term.rows

            # Build entire frame into a buffer, then write atomically
            buf = io.BytesIO()

            # Header
            _draw_header(buf, W, 1, total_steps, "Cameras")

            # Device info
            render_mode = RENDER_MODES[mode_idx]
            _draw_text(buf, 3, 2,
                       f"Camera {i+1}/{len(devices)}: "
                       f"\x1b[96m{dev.label}\x1b[0m  "
                       f"\x1b[90m[{MODE_LABELS[render_mode]}]\x1b[0m")
            _draw_text(buf, 4, 2,
                       f"Type: {dev.dtype}  Device: {dev.device_id}")
            settings_row = 5
            if dev.id_path:
                # Truncate id_path if too long
                id_path_display = dev.id_path if len(dev.id_path) <= W - 14 else dev.id_path[:W - 17] + "..."
                _draw_text(buf, settings_row, 2,
                           f"\x1b[90mID_PATH: {id_path_display}\x1b[0m")
                settings_row += 1

            # Format/mode selection (for v4l2)
            if formats and dev.dtype == "v4l2":
                cur_fmt = formats[format_idx] if format_idx < len(formats) else None
                modes = cur_fmt.modes if cur_fmt else []

                # Format selector
                fmt_str = cur_fmt.fourcc if cur_fmt else "N/A"
                fmt_desc = cur_fmt.description if cur_fmt else ""
                _draw_text(buf, settings_row, 2,
                           f"\x1b[33m[f]\x1b[0m Format: "
                           f"\x1b[1;97m{fmt_str}\x1b[0m "
                           f"\x1b[90m({fmt_desc})\x1b[0m")

                # Mode selector
                if modes and mode_sel_idx < len(modes):
                    cur_mode = modes[mode_sel_idx]
                    mode_str = f"{cur_mode.width}×{cur_mode.height}@{cur_mode.fps}fps"
                else:
                    mode_str = f"{cur_width}×{cur_height}@{cur_fps}fps"
                _draw_text(buf, settings_row + 1, 2,
                           f"\x1b[33m[r]\x1b[0m Resolution: "
                           f"\x1b[1;97m{mode_str}\x1b[0m "
                           f"\x1b[90m({len(modes)} available)\x1b[0m")
                settings_row += 2

            # Depth/IR visualization mode display
            if is_depth_camera:
                depth_mode_str = DEPTH_MODES[depth_mode_idx]
                _draw_text(buf, settings_row, 2,
                           f"\x1b[33m[d]\x1b[0m Visualization: "
                           f"\x1b[1;97m{MODE_LABELS.get(depth_mode_str, depth_mode_str)}\x1b[0m")
                settings_row += 1

            # Format/resolution for RealSense channels
            if formats and dev.dtype.startswith("realsense_"):
                cur_fmt = formats[format_idx] if format_idx < len(formats) else None
                modes = cur_fmt.modes if cur_fmt else []

                # Format selector
                fmt_str = cur_fmt.fourcc if cur_fmt else "N/A"
                fmt_desc = cur_fmt.description if cur_fmt else ""
                _draw_text(buf, settings_row, 2,
                           f"\x1b[33m[f]\x1b[0m Format: "
                           f"\x1b[1;97m{fmt_str}\x1b[0m "
                           f"\x1b[90m({fmt_desc})\x1b[0m")

                # Mode selector
                mode_str = f"{cur_width}×{cur_height}@{cur_fps}fps"
                _draw_text(buf, settings_row + 1, 2,
                           f"\x1b[33m[r]\x1b[0m Resolution: "
                           f"\x1b[1;97m{mode_str}\x1b[0m "
                           f"\x1b[90m({len(modes)} available)\x1b[0m")
                settings_row += 2

            # Live preview
            preview_y = settings_row + 1
            avail_h = max(4, H - preview_y - 5)
            avail_w = W - 4
            preview_w, preview_h = calc_render_size(
                cur_width, cur_height, avail_w, avail_h)

            if cam is not None:
                _, frame = cam.read()
                if frame is not None:
                    # Use appropriate renderer based on device type
                    if is_depth_camera and frame.ndim == 2:
                        # Grayscale/depth frame - use depth renderer
                        depth_mode = DEPTH_MODES[depth_mode_idx]
                        rendered = render_depth(frame, preview_w, preview_h, depth_mode)
                    else:
                        # Color frame - use standard renderer
                        rendered = render_frame(frame, preview_w, preview_h, render_mode)
                    buf.write(blit_frame(rendered, preview_y, 3))
            else:
                _draw_text(buf, preview_y, 3, "(no preview available)")

            prompt_row = preview_y + preview_h + 1

            if phase == "preview":
                # Live-preview phase
                controls = ("\x1b[33m[Enter]\x1b[0m name  "
                            "\x1b[33m[s]\x1b[0m skip  "
                            "\x1b[33m[m]\x1b[0m color  ")
                if formats and dev.dtype == "v4l2":
                    controls += ("\x1b[33m[f]\x1b[0m format  "
                                 "\x1b[33m[r]\x1b[0m res  ")
                if formats and dev.dtype.startswith("realsense_"):
                    controls += ("\x1b[33m[f]\x1b[0m format  "
                                 "\x1b[33m[r]\x1b[0m res  ")
                if is_depth_camera:
                    controls += "\x1b[33m[d]\x1b[0m viz-mode  "
                controls += ("\x1b[33m[\\]\x1b[0m debug  "
                             "\x1b[33m[Esc/q]\x1b[0m quit")
                _draw_text(buf, prompt_row, 2, controls)

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
                elif key == "f" and formats and (dev.dtype == "v4l2" or dev.dtype.startswith("realsense_")):
                    # Open format picker
                    selected = _pick_format(term, out, formats, format_idx, W, H)
                    if selected is not None:
                        format_idx = selected
                        mode_sel_idx = 0
                        cur_fmt = formats[format_idx]
                        cur_format = cur_fmt.fourcc
                        if cur_fmt.modes:
                            m = cur_fmt.modes[0]
                            cur_width, cur_height, cur_fps = m.width, m.height, m.fps
                        _needs_reopen = True
                    _needs_clear = True
                elif key == "r" and formats and (dev.dtype == "v4l2" or dev.dtype.startswith("realsense_")):
                    # Open resolution picker
                    cur_fmt = formats[format_idx] if format_idx < len(formats) else None
                    if cur_fmt and cur_fmt.modes:
                        selected = _pick_resolution(
                            term, out, cur_fmt.modes, mode_sel_idx, W, H)
                        if selected is not None:
                            mode_sel_idx = selected
                            m = cur_fmt.modes[mode_sel_idx]
                            cur_width, cur_height, cur_fps = m.width, m.height, m.fps
                            _needs_reopen = True
                        _needs_clear = True
                elif key == "d" and is_depth_camera:
                    # Cycle depth visualization mode (turbo / gray)
                    depth_mode_idx = (depth_mode_idx + 1) % len(DEPTH_MODES)
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
                # Name-input phase
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
                    phase = "preview"
                    _needs_clear = True
                else:
                    chosen_name = result

        if cam:
            cam.close()

        if chosen_name:
            # Map realsense_* types back to "realsense" and extract channel
            cam_type = dev.dtype
            channel = "color"
            if dev.dtype.startswith("realsense_"):
                cam_type = "realsense"
                channel = dev.properties.get("channel", dev.dtype.replace("realsense_", ""))

            configs.append(CameraConfig(
                name=chosen_name,
                type=cam_type,
                device=dev.device_id,
                width=cur_width,
                height=cur_height,
                fps=cur_fps,
                pixel_format=cur_format,
                id_path=dev.id_path,
                channel=channel,
            ))

    return configs


def _get_airbot_robot(dev: DetectedDevice):
    """Get an AIRBOTPlay instance for device identification."""
    if dev.dtype != "airbot_play":
        return None
    try:
        from rollio.robot import AIRBOTPlay, is_airbot_available
        if not is_airbot_available():
            return None
        can_interface = dev.properties.get("can_interface", str(dev.device_id))
        return AIRBOTPlay(can_interface=can_interface)
    except Exception:
        return None


def _screen_robots(term: _Term, out, devices: list[DetectedDevice]
                   , total_steps: int = 5) -> list[RobotConfig]:
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
        
        # Start identification for AIRBOT devices
        airbot_robot = _get_airbot_robot(dev)
        if airbot_robot is not None:
            try:
                airbot_robot.identify_start()
            except Exception:
                pass

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
            elif dev.dtype == "airbot_play" and airbot_robot is not None:
                _draw_text(buf, 5, 2,
                           "\x1b[93m⟳ LED blinking orange + gravity compensation "
                           "(move arm to identify)\x1b[0m")

            # Robot state display (re-read every iteration)
            pos = None
            n_joints = dev.properties.get("num_joints", 6)
            
            # Try AIRBOT robot first (for real hardware)
            if airbot_robot is not None and airbot_robot._is_open:
                try:
                    # Step gravity compensation to keep robot in free drive
                    airbot_robot.step_free_drive()
                    
                    joint_state = airbot_robot.read_joint_state()
                    if joint_state.is_valid and joint_state.position is not None:
                        pos = joint_state.position
                        n_joints = len(pos)
                except Exception:
                    pass
            
            # Fall back to legacy sensor interface (for pseudo robot)
            if pos is None and rob is not None:
                _, state = rob.read()
                pos = state.get("position", np.zeros(0))
                n_joints = len(pos) if len(pos) > 0 else n_joints
            
            # Display joint positions
            if pos is not None and len(pos) > 0:
                start_row = 7
                bar_w = min(40, W - 25)
                for j in range(n_joints):
                    p = pos[j] if j < len(pos) else 0.0
                    frac = float(np.clip((p + 2) / 4, 0, 1))
                    bar_len = int(frac * bar_w)
                    bar = "█" * bar_len + "░" * (bar_w - bar_len)
                    # For pseudo robot, mark last joint as moving
                    marker = ""
                    if dev.dtype == "pseudo" and j == n_joints - 1:
                        marker = " \x1b[91m← moving\x1b[0m"
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
                    break               # skip this robot (identify_stop called below)
                elif key == "\\":
                    show_debug = not show_debug
                elif key == "q" or key == "\x1b":
                    # Stop identification before quitting
                    if airbot_robot is not None:
                        try:
                            airbot_robot.identify_stop()
                        except Exception:
                            pass
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

        # Stop identification for AIRBOT devices
        if airbot_robot is not None:
            try:
                airbot_robot.identify_stop()
            except Exception:
                pass

        if rob:
            rob.close()

        if chosen_name:
            configs.append(RobotConfig(
                name=chosen_name,
                type=dev.dtype,
                role=chosen_role,
                num_joints=dev.properties.get("num_joints", 6),
                device=str(dev.properties.get("can_interface", dev.device_id)),
            ))

    return configs


def _screen_settings(
    term: _Term,
    out,
    robots: list[RobotConfig],
    *,
    step: int = 3,
    total_steps: int = 5,
) -> tuple[str, str, str, str, str] | None:
    """Project/settings screen — project, storage, mode, and codecs."""
    W, H = term.cols, term.rows

    out.write(_SY_S + b"\x1b[2J")
    _draw_header(out, W, step, total_steps, "Project Settings")

    _draw_text(out, 4, 2,
               "Configure project, collection mode, and codecs.")
    out.write(_SY_E)
    out.flush()

    name = _prompt_line(term, out, 6, 2, "Project name: ", "default")
    if name is None:
        return None

    storage = _prompt_line(term, out, 8, 2, "Storage root: ", "~/rollio_data")
    if storage is None:
        return None

    mode_idx = _pick_option(
        term,
        out,
        title="COLLECTION MODE",
        options=[label for _, label in MODE_OPTIONS],
        current_idx=0,
        subtitle="Choose the collection workflow to configure.",
    )
    if mode_idx is None:
        return None
    mode = MODE_OPTIONS[mode_idx][0]
    if mode == "teleop":
        warning_lines = _teleop_warning_lines(robots)
        if warning_lines is not None:
            _show_warning_screen(
                term,
                out,
                title="Tele-op Requirements",
                lines=warning_lines,
                prompt="Press any key to cancel setup.",
                step=step,
                total_steps=total_steps,
            )
            return None

    rgb_options = list(available_rgb_codec_options())
    rgb_idx = _pick_option(
        term,
        out,
        title="RGB CODEC",
        options=[option.label for option in rgb_options],
        current_idx=0,
        subtitle="Select the codec used for RGB camera streams.",
    )
    if rgb_idx is None:
        return None
    rgb_codec = rgb_options[rgb_idx].name

    depth_options = list(available_depth_codec_options())
    depth_idx = _pick_option(
        term,
        out,
        title="DEPTH CODEC",
        options=[option.label for option in depth_options],
        current_idx=0,
        subtitle="Select the codec used for depth/grayscale streams.",
    )
    if depth_idx is None:
        return None
    depth_codec = depth_options[depth_idx].name

    return name, storage, mode, rgb_codec, depth_codec


def _screen_teleop_pairs(
    term: _Term,
    out,
    robots: list[RobotConfig],
    *,
    step: int = 4,
    total_steps: int = 5,
) -> list[TeleopPairConfig] | None:
    """Configure explicit tele-op pairings for leader/follower robots."""
    warning_lines = _teleop_warning_lines(robots)
    if warning_lines is not None:
        _show_warning_screen(
            term,
            out,
            title="Tele-op Pairing",
            lines=warning_lines,
            prompt="Press any key to cancel setup.",
            step=step,
            total_steps=total_steps,
        )
        return None
    leaders = [robot for robot in robots if robot.role == "leader"]
    followers = [robot for robot in robots if robot.role == "follower"]

    suggested = suggest_teleop_pairs(robots)
    pairs: list[TeleopPairConfig] = []
    remaining_leaders = list(leaders)
    remaining_followers = list(followers)

    for idx in range(min(len(leaders), len(followers))):
        default_pair = suggested[idx] if idx < len(suggested) else None
        W = term.cols
        out.write(_SY_S + b"\x1b[2J")
        _draw_header(out, W, step, total_steps, f"Tele-op Pairing ({idx + 1}/{min(len(leaders), len(followers))})")
        _draw_text(
            out,
            4,
            2,
            "Select which leader controls which follower, and how the mapping works.",
        )
        out.write(_SY_E)
        out.flush()

        follower_default_idx = 0
        if default_pair is not None:
            for fi, follower in enumerate(remaining_followers):
                if follower.name == default_pair.follower:
                    follower_default_idx = fi
                    break
        follower_idx = _pick_option(
            term,
            out,
            title=f"PAIR {idx + 1} — FOLLOWER",
            options=[f"{robot.name} ({robot.type})" for robot in remaining_followers],
            current_idx=follower_default_idx,
            subtitle="Choose the follower arm to be controlled in this pair.",
        )
        if follower_idx is None:
            return None
        follower = remaining_followers.pop(follower_idx)

        leader_default_idx = 0
        if default_pair is not None:
            for li, leader in enumerate(remaining_leaders):
                if leader.name == default_pair.leader:
                    leader_default_idx = li
                    break
        leader_idx = _pick_option(
            term,
            out,
            title=f"PAIR {idx + 1} — LEADER",
            options=[f"{robot.name} ({robot.type})" for robot in remaining_leaders],
            current_idx=leader_default_idx,
            subtitle="Choose the leader arm that drives the selected follower.",
        )
        if leader_idx is None:
            return None
        leader = remaining_leaders.pop(leader_idx)

        mapper_options = [label for _, label in MAPPER_OPTIONS]
        default_mapper = default_mapper_for_pair(leader, follower)
        mapper_default_idx = 0 if default_mapper == "joint_direct" else 1
        mapper_idx = _pick_option(
            term,
            out,
            title=f"PAIR {idx + 1} — MAPPING",
            options=mapper_options,
            current_idx=mapper_default_idx,
            subtitle="Choose direct joint mapping or FK-IK pose mapping for this pair.",
        )
        if mapper_idx is None:
            return None
        mapper = MAPPER_OPTIONS[mapper_idx][0]

        pairs.append(TeleopPairConfig(
            name=f"pair_{idx}",
            leader=leader.name,
            follower=follower.name,
            mapper=mapper,
        ))

    validate_teleop_pairs(robots, pairs)
    return pairs


def _camera_types_match(dev_dtype: str, cfg_type: str, cfg_channel: str) -> bool:
    """Check if a detected camera matches a config entry."""
    if dev_dtype == cfg_type:
        return True
    if cfg_type == "realsense" and dev_dtype.startswith("realsense_"):
        return dev_dtype.replace("realsense_", "") == cfg_channel
    return False


def _match_camera_devices(
    cam_configs: list[CameraConfig],
    cam_devs: list[DetectedDevice],
) -> list[tuple[CameraConfig, DetectedDevice | None]]:
    """Match configured cameras to detected devices for preview metadata."""
    matched: list[tuple[CameraConfig, DetectedDevice | None]] = []
    used_cam_devs: set[int] = set()
    for cc in cam_configs:
        matched_dev = None
        for di, dev in enumerate(cam_devs):
            if (
                di not in used_cam_devs
                and str(dev.device_id) == str(cc.device)
                and _camera_types_match(dev.dtype, cc.type, cc.channel)
            ):
                matched_dev = dev
                used_cam_devs.add(di)
                break
        if matched_dev is None:
            for di, dev in enumerate(cam_devs):
                if di not in used_cam_devs and _camera_types_match(dev.dtype, cc.type, cc.channel):
                    matched_dev = dev
                    used_cam_devs.add(di)
                    break
        matched.append((cc, matched_dev))
    return matched


def _match_robot_devices(
    rob_configs: list[RobotConfig],
    rob_devs: list[DetectedDevice],
) -> list[tuple[RobotConfig, DetectedDevice | None]]:
    """Match configured robots to detected devices for preview metadata."""
    matched: list[tuple[RobotConfig, DetectedDevice | None]] = []
    used_rob_devs: set[int] = set()
    for rc in rob_configs:
        matched_dev = None
        for di, dev in enumerate(rob_devs):
            if di not in used_rob_devs and str(dev.properties.get("can_interface", dev.device_id)) == str(rc.device):
                matched_dev = dev
                used_rob_devs.add(di)
                break
        if matched_dev is None:
            for di, dev in enumerate(rob_devs):
                if di not in used_rob_devs and dev.dtype == rc.type:
                    matched_dev = dev
                    used_rob_devs.add(di)
                    break
        matched.append((rc, matched_dev))
    return matched


def _screen_summary(term: _Term, out,
                    cam_configs: list[CameraConfig],
                    rob_configs: list[RobotConfig],
                    cam_devs: list[DetectedDevice],
                    rob_devs: list[DetectedDevice],
                    project_name: str,
                    storage_root: str,
                    output_path: str,
                    *,
                    mode: str,
                    video_codec: str,
                    depth_codec: str,
                    teleop_pairs: list[TeleopPairConfig],
                    step: int = 5,
                    total_steps: int = 5) -> bool:
    """Live summary screen with camera previews + robot states.

    Returns True to save, False to cancel.
    """
    cam_entries = _match_camera_devices(cam_configs, cam_devs)
    rob_entries = _match_robot_devices(rob_configs, rob_devs)

    mode_idx = 0  # "true" (24-bit truecolor) for best preview quality
    show_debug = False
    _t_prev = time.monotonic()
    _fps = 0.0
    result = None
    _needs_clear = True
    selected_cam = 0 if cam_entries else -1
    _needs_restart = True
    preview_runtime: AsyncCollectionRuntime | None = None

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

    def _build_preview_runtime() -> AsyncCollectionRuntime:
        preview_fps = max([30, *[cfg.fps for cfg in cam_configs]])
        preview_cfg = RollioConfig(
            project_name=project_name,
            fps=preview_fps,
            mode=mode,
            cameras=cam_configs,
            robots=rob_configs,
            teleop_pairs=teleop_pairs,
            storage=StorageConfig(root=storage_root),
            encoder=EncoderConfig(
                video_codec=video_codec,
                depth_codec=depth_codec,
            ),
        )
        return AsyncCollectionRuntime.from_config(
            preview_cfg,
            scheduler_driver="asyncio",
        )

    try:
        while result is None:
            if _needs_restart:
                if preview_runtime is not None:
                    preview_runtime.close()
                preview_runtime = _build_preview_runtime()
                preview_runtime.open()
                _needs_clear = True
                _needs_restart = False

            W, H = term.cols, term.rows
            buf = io.BytesIO()
            latest_frames = preview_runtime.latest_frames() if preview_runtime is not None else {}
            latest_robot_states = (
                preview_runtime.latest_robot_states()
                if preview_runtime is not None
                else {}
            )

            # FPS tracking
            _t_now = time.monotonic()
            _dt = _t_now - _t_prev
            _t_prev = _t_now
            if _dt > 0:
                _fps = 0.9 * _fps + 0.1 / _dt

            _draw_header(buf, W, step, total_steps, "Summary — Live Preview")

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
            _draw_text_clear(buf, row, 2,
                             f"│ \x1b[1mMode:\x1b[0m \x1b[97;1m{mode}\x1b[0m",
                             info_w)
            row += 1
            _draw_text_clear(buf, row, 2,
                             f"│ \x1b[1mRGB codec:\x1b[0m \x1b[90m{video_codec[:info_w-16]}\x1b[0m",
                             info_w)
            row += 1
            _draw_text_clear(buf, row, 2,
                             f"│ \x1b[1mDepth codec:\x1b[0m \x1b[90m{depth_codec[:info_w-18]}\x1b[0m",
                             info_w)
            row += 1
            _draw_text_clear(buf, row, 2, "├" + box_line + "┤", info_w)
            row += 1
            _draw_text_clear(buf, row, 2,
                             f"│ \x1b[1;96mCameras ({len(cam_configs)})\x1b[0m "
                             f"\x1b[90m[1-{len(cam_configs)}] select\x1b[0m"
                             if cam_configs else
                             f"│ \x1b[1;96mCameras (0)\x1b[0m",
                             info_w)
            row += 1
            for ci, (cc, _) in enumerate(cam_entries):
                is_sel = (ci == selected_cam)
                sel_mark = "\x1b[97;1m▸\x1b[0m" if is_sel else " "
                highlight = "\x1b[97;1;44m" if is_sel else "\x1b[96m"
                # Line 1: index and name
                _draw_text_clear(buf, row, 2,
                                 f"│ {sel_mark}{ci+1}.{highlight}{cc.name[:12]}\x1b[0m",
                                 info_w)
                row += 1
                # Line 2: type | format | resolution | fps (aligned)
                type_str = cc.type
                if cc.type == "realsense":
                    # Abbreviate channel names: color→col, depth→dep, infrared→ir
                    ch_abbr = {"color": "col", "depth": "dep", "infrared": "ir"}.get(cc.channel, cc.channel or "col")
                    type_str = f"rs:{ch_abbr}"
                fmt_str = cc.pixel_format or "?"
                res_str = f"{cc.width}×{cc.height}"
                fps_str = f"{cc.fps}fps" if cc.fps else "?fps"
                _draw_text_clear(buf, row, 2,
                                 f"│    \x1b[90m{type_str:<8} {fmt_str:<5} {res_str:<10} {fps_str}\x1b[0m",
                                 info_w)
                row += 1
            _draw_text_clear(buf, row, 2, "├" + box_line + "┤", info_w)
            row += 1
            _draw_text_clear(buf, row, 2,
                             f"│ \x1b[1;93mRobots ({len(rob_configs)})\x1b[0m",
                             info_w)
            row += 1
            for rc, matched_rob_dev in rob_entries:
                role_clr = "92" if rc.role == "leader" else "33"
                _draw_text_clear(buf, row, 2,
                                 f"│  \x1b[93m{rc.name[:12]:<12}\x1b[0m "
                                 f"\x1b[{role_clr}m{rc.role}\x1b[0m "
                                 f"\x1b[90m{rc.num_joints}-DOF\x1b[0m",
                                 info_w)
                row += 1
                # Show SN for AIRBOT devices
                if rc.type == "airbot_play" and matched_rob_dev is not None:
                    sn = matched_rob_dev.properties.get("serial_number", "")
                    eef = matched_rob_dev.properties.get("end_effector_type", "")
                    info_line = ""
                    if sn:
                        info_line += f"SN:{sn}"
                    if eef:
                        if info_line:
                            info_line += " "
                        info_line += f"EEF:{eef}"
                    if info_line:
                        _draw_text_clear(buf, row, 2,
                                         f"│     \x1b[90m{info_line}\x1b[0m",
                                         info_w)
                        row += 1
            if teleop_pairs:
                _draw_text_clear(buf, row, 2, "├" + box_line + "┤", info_w)
                row += 1
                _draw_text_clear(buf, row, 2,
                                 f"│ \x1b[1;95mTele-op pairs ({len(teleop_pairs)})\x1b[0m",
                                 info_w)
                row += 1
                for pair in teleop_pairs:
                    mapper_label = "direct" if pair.mapper == "joint_direct" else "FK-IK"
                    pair_text = f"│  \x1b[95m{pair.leader}\x1b[0m → \x1b[93m{pair.follower}\x1b[0m"
                    _draw_text_clear(buf, row, 2, pair_text, info_w)
                    row += 1
                    _draw_text_clear(buf, row, 2,
                                     f"│     \x1b[90m{mapper_label}\x1b[0m", info_w)
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
            n_cams = len(cam_entries)
            n_robs = len(rob_entries)

            # Calculate space for cameras and robots
            avail_h = max(4, H - 6)
            cam_h_total = max(4, (avail_h * 2) // 3) if n_cams else 0
            rob_h_total = avail_h - cam_h_total if n_robs else 0

            # Draw camera previews in a grid
            if n_cams > 0:
                mode = RENDER_MODES[mode_idx]
                cam_title = "─── CAMERAS "
                cam_title += "─" * max(0, preview_w - len(cam_title))
                _draw_text(buf, preview_row, preview_col,
                           f"\x1b[1;96m{cam_title[:preview_w]}\x1b[0m")
                preview_row += 1

                # Calculate grid dimensions
                # Try to make cells roughly square-ish, max 3 columns
                if n_cams <= 2:
                    grid_cols = n_cams
                elif n_cams <= 4:
                    grid_cols = 2
                elif n_cams <= 6:
                    grid_cols = 3
                else:
                    grid_cols = min(4, (n_cams + 1) // 2)
                grid_rows = (n_cams + grid_cols - 1) // grid_cols

                # Calculate cell size (leave 2 rows for label per cell)
                cell_w = max(15, (preview_w - 2) // grid_cols)
                cell_h = max(4, cam_h_total // grid_rows)
                preview_h_per = max(2, cell_h - 2)  # leave room for label

                for ci, (cc, _) in enumerate(cam_entries):
                    # Calculate grid position
                    grid_r = ci // grid_cols
                    grid_c = ci % grid_cols
                    cell_row = preview_row + grid_r * cell_h
                    cell_col = preview_col + grid_c * cell_w

                    is_sel = (ci == selected_cam)
                    frame = latest_frames.get(cc.name)
                    if frame is not None:
                        try:
                            fh, fw = frame.shape[:2]
                            rw, rh = calc_render_size(
                                fw, fh, cell_w - 2, preview_h_per)
                            if cc.channel in ("depth", "infrared") and frame.ndim == 2:
                                rendered = render_depth(frame, rw, rh, "turbo")
                            else:
                                rendered = render_frame(frame, rw, rh, mode)
                            buf.write(blit_frame(rendered, cell_row, cell_col + 1))
                        except Exception:
                            _draw_text(buf, cell_row, cell_col + 1,
                                       f"\x1b[90m(err)\x1b[0m")
                    else:
                        _draw_text(buf, cell_row, cell_col + 1,
                                   f"\x1b[90m(no preview)\x1b[0m")

                    # Draw camera label below preview
                    label_row = cell_row + preview_h_per
                    sel_ind = "\x1b[97;1m▸\x1b[0m" if is_sel else " "
                    # Compact label with channel info for realsense
                    type_str = cc.type
                    if cc.type == "realsense" and cc.channel:
                        type_str = f"rs:{cc.channel[:3]}"
                    label = f"{sel_ind}{ci+1}.\x1b[96;1m{cc.name[:6]}\x1b[0m \x1b[90m{type_str}\x1b[0m"
                    _draw_text_clear(buf, label_row, cell_col, label, cell_w)

                # Move preview_row past the grid
                preview_row += grid_rows * cell_h

            # Draw robot states
            if n_robs > 0:
                rob_title = "─── ROBOTS "
                rob_title += "─" * max(0, preview_w - len(rob_title))
                _draw_text(buf, preview_row, preview_col,
                           f"\x1b[1;93m{rob_title[:preview_w]}\x1b[0m")
                preview_row += 1
                bar_w = min(30, preview_w - 20)

                for rc, _ in rob_entries:
                    role_clr = "92" if rc.role == "leader" else "33"
                    _draw_text(buf, preview_row, preview_col,
                               f"\x1b[{role_clr};1m{rc.name}\x1b[0m "
                               f"\x1b[90m({rc.num_joints}-DOF, {rc.role})\x1b[0m")
                    try:
                        state = latest_robot_states.get(rc.name, {})
                        pos = state.get("position", np.zeros(0))
                        for j in range(len(pos)):
                            p = pos[j]
                            frac = float(np.clip((p + 2) / 4, 0, 1))
                            bar_len = int(frac * bar_w)
                            bar = "█" * bar_len + "░" * (bar_w - bar_len)
                            _draw_text(buf, preview_row + 1 + j,
                                       preview_col + 2,
                                       f"j{j} \x1b[36m{p:+5.2f}\x1b[0m "
                                       f"\x1b[33m{bar}\x1b[0m")
                        preview_row += 1 + len(pos) + 1
                    except Exception:
                        _draw_text(buf, preview_row + 1, preview_col + 2,
                                   "\x1b[90m(error)\x1b[0m")
                        preview_row += 3

            # Debug overlay
            if show_debug:
                _draw_text(buf, 2, W - 18,
                           f"\x1b[48;5;234m\x1b[38;5;82m FPS: {_fps:5.1f} \x1b[0m")

            # ── Footer / controls ──
            footer_row = H - 2
            cam_controls = ""
            if selected_cam >= 0 and cam_entries[selected_cam][1]:
                dev = cam_entries[selected_cam][1]
                if dev and dev.formats and (dev.dtype == "v4l2" or dev.dtype.startswith("realsense_")):
                    cam_controls = "\x1b[33m[f]\x1b[0m format  \x1b[33m[r]\x1b[0m resolution  "
            _draw_text_clear(buf, footer_row, 2,
                             f"\x1b[33m[Enter]\x1b[0m save  "
                             f"{cam_controls}"
                             "\x1b[33m[m]\x1b[0m mode  "
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
            # Camera selection with number keys
            elif key and key.isdigit():
                idx = int(key) - 1
                if 0 <= idx < len(cam_entries):
                    selected_cam = idx
            # Format selection for selected camera
            elif key == "f" and selected_cam >= 0:
                cc, dev = cam_entries[selected_cam]
                if dev and dev.formats and (dev.dtype == "v4l2" or dev.dtype.startswith("realsense_")):
                    # Find current format index
                    cur_fmt_idx = 0
                    for fi, fmt in enumerate(dev.formats):
                        if fmt.fourcc == cc.pixel_format:
                            cur_fmt_idx = fi
                            break
                    new_idx = _pick_format(term, out, dev.formats, cur_fmt_idx, W, H)
                    if new_idx is not None and new_idx != cur_fmt_idx:
                        new_fmt = dev.formats[new_idx]
                        cc.pixel_format = new_fmt.fourcc
                        # Pick first mode of new format
                        if new_fmt.modes:
                            m = new_fmt.modes[0]
                            cc.width, cc.height, cc.fps = m.width, m.height, m.fps
                        _needs_restart = True
                    _needs_clear = True
            # Resolution selection for selected camera
            elif key == "r" and selected_cam >= 0:
                cc, dev = cam_entries[selected_cam]
                if dev and dev.formats and (dev.dtype == "v4l2" or dev.dtype.startswith("realsense_")):
                    # Find current format and mode index
                    cur_fmt = None
                    for fmt in dev.formats:
                        if fmt.fourcc == cc.pixel_format:
                            cur_fmt = fmt
                            break
                    if cur_fmt and cur_fmt.modes:
                        cur_mode_idx = 0
                        for mi, m in enumerate(cur_fmt.modes):
                            if m.width == cc.width and m.height == cc.height and m.fps == cc.fps:
                                cur_mode_idx = mi
                                break
                        new_idx = _pick_resolution(term, out, cur_fmt.modes, cur_mode_idx, W, H)
                        if new_idx is not None:
                            m = cur_fmt.modes[new_idx]
                            cc.width, cc.height, cc.fps = m.width, m.height, m.fps
                            _needs_restart = True
                        _needs_clear = True

            # Throttle to ~30 FPS
            _el = time.monotonic() - _t_now
            if _el < 0.033:
                time.sleep(0.033 - _el)

    finally:
        if preview_runtime is not None:
            preview_runtime.close()

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Main wizard entry point
# ═══════════════════════════════════════════════════════════════════════

def run_wizard(
    output_path: str,
    *,
    simulated_cameras: int = 0,
    simulated_arms: int = 0,
) -> RollioConfig | None:
    """Run the interactive setup wizard.  Returns config or None on abort."""
    print("Scanning for hardware…")
    cam_devs = scan_cameras(
        include_simulated=simulated_cameras > 0,
        simulated_count=simulated_cameras,
    )
    rob_devs = scan_robots(
        include_simulated=simulated_arms > 0,
        simulated_count=simulated_arms,
    )
    print(f"  Found {len(cam_devs)} camera(s), {len(rob_devs)} robot(s).")
    print("Launching wizard TUI…")
    time.sleep(0.5)

    out = sys.stdout.buffer
    total_steps = 5

    with _Term() as term:
        # Step 1: Cameras
        cam_configs = _screen_cameras(term, out, cam_devs, total_steps=total_steps)

        # Step 2: Robots
        rob_configs = _screen_robots(term, out, rob_devs, total_steps=total_steps)

        # Step 3: Project/settings
        settings = _screen_settings(
            term,
            out,
            rob_configs,
            step=3,
            total_steps=total_steps,
        )
        if settings is None:
            return None
        project_name, storage_root, mode, video_codec, depth_codec = settings

        teleop_pairs: list[TeleopPairConfig] = []
        if mode == "teleop":
            pair_configs = _screen_teleop_pairs(
                term,
                out,
                rob_configs,
                step=4,
                total_steps=total_steps,
            )
            if pair_configs is None:
                return None
            teleop_pairs = pair_configs

        # Step 5: Live summary screen with previews
        should_save = _screen_summary(
            term, out,
            cam_configs, rob_configs,
            cam_devs, rob_devs,
            project_name, storage_root,
            output_path,
            mode=mode,
            video_codec=video_codec,
            depth_codec=depth_codec,
            teleop_pairs=teleop_pairs,
            step=5,
            total_steps=total_steps,
        )

        if not should_save:
            return None

    return RollioConfig(
        project_name=project_name,
        mode=mode,
        cameras=cam_configs,
        robots=rob_configs,
        teleop_pairs=teleop_pairs,
        storage=StorageConfig(root=storage_root),
        encoder=EncoderConfig(
            video_codec=video_codec,
            depth_codec=depth_codec,
        ),
    )
