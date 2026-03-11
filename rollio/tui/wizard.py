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
import threading
import time
import tty
from collections import deque

import numpy as np

from rollio.collect import AsyncCollectionRuntime
from rollio.config.pairing import (
    default_mapper_for_pair,
    suggest_teleop_pairs,
    validate_teleop_pairs,
)
from rollio.config.schema import (
    CameraConfig,
    RobotConfig,
    RollioConfig,
    StorageConfig,
    TeleopPairConfig,
    EncoderConfig,
)
from rollio.episode.codecs import (
    available_depth_codec_options,
    available_rgb_codec_options,
)
from rollio.sensors.base import CameraFormat, CameraMode, ImageSensor
from rollio.sensors.pseudo_camera import PseudoCamera
from rollio.tui.timing import build_timing_panel_lines, make_timing_trace
from rollio.sensors.pseudo_robot import PseudoRobot
from rollio.sensors.realsense_camera import RealSenseCamera
from rollio.sensors.scanner import DetectedDevice, scan_cameras, scan_robots
from rollio.sensors.v4l2_camera import V4L2Camera
from rollio.tui.renderer import (
    RENDER_MODES,
    DEPTH_MODES,
    MODE_LABELS,
    blit_frame,
    calc_render_size,
    render_frame,
    render_depth,
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
_LOADING_FRAMES = ("|", "/", "-", "\\")


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

    def _read_ready_char(self, timeout: float) -> str | None:
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None

    def _decode_key(self, first_char: str, sequence_timeout: float = 0.02) -> str:
        if first_char != "\x1b":
            return first_char

        sequence = first_char
        while True:
            timeout = sequence_timeout if len(sequence) == 1 else 0.002
            extra = self._read_ready_char(timeout)
            if extra is None:
                break
            sequence += extra
            if sequence.startswith("\x1b[") or sequence.startswith("\x1bO"):
                if extra.isalpha() or extra == "~":
                    break
            elif len(sequence) > 1:
                break

        if sequence == "\x1b":
            return "\x1b"
        if sequence.startswith("\x1b[") or sequence.startswith("\x1bO"):
            final = sequence[-1]
            if final == "A":
                return "UP"
            if final == "B":
                return "DOWN"
            if final == "C":
                return "RIGHT"
            if final == "D":
                return "LEFT"
        return sequence

    def read_key(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            return self._decode_key(ch)
        return None

    def read_key_blocking(self, timeout: float = 0.05) -> str | None:
        if select.select([sys.stdin], [], [], timeout)[0]:
            return self._decode_key(sys.stdin.read(1))
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Helper: open a temp sensor for preview
# ═══════════════════════════════════════════════════════════════════════


def _make_camera(
    dev: DetectedDevice,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    pixel_format: str | None = None,
) -> ImageSensor | None:
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
                width=w,
                height=h,
                fps=f,
                pixel_format=pf,
            )
            cam.open()
            return cam
        except (OSError, RuntimeError, ValueError, TypeError):
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
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
    return None


def _make_robot(dev: DetectedDevice) -> PseudoRobot | None:
    if dev.dtype == "pseudo":
        rob = PseudoRobot(name="preview", n_joints=dev.properties.get("num_joints", 6))
        rob.open()
        return rob
    return None


def _default_robot_name(dtype: str, index: int) -> str:
    prefix = {
        "airbot_play": "arm",
        "airbot_e2b": "eef_e2b",
        "airbot_g2": "eef_g2",
        "pseudo": "arm",
    }.get(dtype, "robot")
    return f"{prefix}_{index}"


def _default_robot_role(dtype: str) -> str:
    if dtype == "airbot_e2b":
        return "leader"
    return "follower"


# ═══════════════════════════════════════════════════════════════════════
#  Wizard screens
# ═══════════════════════════════════════════════════════════════════════


def _draw_header(out, W: int, step: int, total: int, title: str):
    """Draw wizard header bar."""
    hdr = f"  ROLLIO SETUP WIZARD  —  Step {step}/{total}: {title}  "
    out.write(f"\x1b[1;1H\x1b[48;5;24m\x1b[97;1m{hdr.ljust(W)}\x1b[0m".encode())


def _draw_text(out, row: int, col: int, text: str):
    out.write(f"\x1b[{row};{col}H{text}\x1b[0m".encode())


def _draw_loading_screen(
    term: _Term,
    out,
    *,
    step: int,
    total: int,
    title: str,
    message: str,
    frame_idx: int,
) -> None:
    W = term.cols
    spinner = _LOADING_FRAMES[frame_idx % len(_LOADING_FRAMES)]
    out.write(_SY_S + b"\x1b[2J")
    _draw_header(out, W, step, total, f"{title} {spinner}")
    _draw_text(out, 4, 2, message)
    out.write(_SY_E)
    out.flush()


def _show_loading_transition(
    term: _Term,
    out,
    *,
    step: int,
    total: int,
    title: str,
    message: str,
    duration: float = 0.3,
) -> None:
    end_time = time.monotonic() + max(0.0, duration)
    frame_idx = 0
    while True:
        _draw_loading_screen(
            term,
            out,
            step=step,
            total=total,
            title=title,
            message=message,
            frame_idx=frame_idx,
        )
        if time.monotonic() >= end_time:
            return
        frame_idx += 1
        time.sleep(0.08)


def _run_with_loading(
    term: _Term,
    out,
    *,
    step: int,
    total: int,
    title: str,
    message: str,
    work,
):
    result: dict[str, object] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            result["value"] = work()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            result["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_worker, name="rollio-wizard-loading", daemon=True)
    thread.start()

    frame_idx = 0
    while True:
        _draw_loading_screen(
            term,
            out,
            step=step,
            total=total,
            title=title,
            message=message,
            frame_idx=frame_idx,
        )
        if done.wait(0.08):
            break
        frame_idx += 1

    thread.join()
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result.get("value")


def _format_joint_preview(dtype: str, value: float) -> tuple[str, float]:
    if dtype in {"airbot_e2b", "airbot_g2"}:
        frac = float(np.clip(value / 0.07, 0.0, 1.0))
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


def _airbot_led_block(blink_on: bool | None = None, width: int = 18) -> str:
    """Render the AIRBOT identification LED as a blinking orange block."""
    if blink_on is None:
        blink_on = (time.monotonic() * 4.0) % 1.0 < 0.5
    if blink_on:
        return f"\x1b[5m\x1b[48;5;208m{' ' * width}\x1b[0m"
    return f"\x1b[48;5;236m{' ' * width}\x1b[0m"


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


def _prompt_line(
    term: _Term, out, row: int, col: int, prompt: str, default: str = ""
) -> str | None:
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
        elif key == "\x1b":  # Escape
            return None
        elif key == "\x7f" or key == "\x08":  # Backspace
            if buf:
                buf.pop()
        elif len(key) == 1 and key.isprintable():
            buf.append(key)


def _pick_format(
    term: _Term, out, formats: list[CameraFormat], current_idx: int, W: int, _H: int
) -> int | None:
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
        buf.write(
            f"\x1b[1;1H\x1b[48;5;24m\x1b[97;1m{'  SELECT FORMAT  ':<{W}}\x1b[0m".encode()
        )

        # Instructions
        _draw_text(
            buf,
            3,
            2,
            "Use ↑/↓, or enter number and press Enter, or \x1b[33m[Esc]\x1b[0m to cancel",
        )
        _draw_text(buf, 4, 2, f"Input: \x1b[1;97m{input_buf}\x1b[5m_\x1b[0m")

        # List formats
        start_row = 6
        for idx, fmt in enumerate(formats):
            is_cur = idx == current_idx
            num_str = f"{idx + 1:>2}"
            n_modes = len(fmt.modes)
            if is_cur:
                text = (
                    f"\x1b[1;92m[{num_str}] {fmt.fourcc}\x1b[0m "
                    f"\x1b[90m({fmt.description}, {n_modes} modes)\x1b[0m"
                )
            else:
                text = (
                    f"\x1b[33m[{num_str}]\x1b[0m {fmt.fourcc} "
                    f"\x1b[90m({fmt.description}, {n_modes} modes)\x1b[0m"
                )
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
            else:
                return current_idx
            input_buf = ""  # Clear on invalid
        elif key == "\x7f" or key == "\x08":  # Backspace
            input_buf = input_buf[:-1]
        elif key == "UP" or key == "LEFT":
            input_buf = ""
            current_idx = max(0, current_idx - 1)
        elif key == "DOWN" or key == "RIGHT":
            input_buf = ""
            current_idx = min(len(formats) - 1, current_idx + 1)
        elif key.isdigit():
            input_buf += key


def _pick_resolution(
    term: _Term, out, modes: list[CameraMode], current_idx: int, W: int, H: int
) -> int | None:
    """Display resolution picker grid.

    Each item is a separate W×H@FPS tuplet (standard v4l2 form).
    Returns selected index, or None if cancelled (Esc).
    """
    # Sort modes: by resolution (largest first), then by FPS (highest first)
    sorted_modes = sorted(
        modes, key=lambda m: (m.width * m.height, m.fps), reverse=True
    )

    # Build display items: (idx_in_original_modes, display_str)
    items: list[tuple[int, str]] = []
    for m in sorted_modes:
        orig_idx = modes.index(m)
        display = f"{m.width}×{m.height}@{m.fps}"
        items.append((orig_idx, display))

    # Calculate grid layout
    max_item_w = max(len(it[1]) for it in items) + 6  # "[NN] " + padding
    cols = max(1, (W - 4) // max_item_w)
    rows = (len(items) + cols - 1) // cols

    input_buf = ""
    scroll_offset = 0
    visible_rows = max(4, H - 8)
    current_display_idx = 0
    for idx, (orig_idx, _display) in enumerate(items):
        if orig_idx == current_idx:
            current_display_idx = idx
            break

    def _clamp_visible_row() -> None:
        nonlocal scroll_offset
        current_row = current_display_idx // cols
        max_scroll = max(0, rows - visible_rows)
        if current_row < scroll_offset:
            scroll_offset = current_row
        elif current_row >= scroll_offset + visible_rows:
            scroll_offset = current_row - visible_rows + 1
        scroll_offset = max(0, min(scroll_offset, max_scroll))

    while True:
        buf = io.BytesIO()
        buf.write(b"\x1b[2J\x1b[H")

        # Header
        buf.write(
            f"\x1b[1;1H\x1b[48;5;24m\x1b[97;1m{'  SELECT RESOLUTION  ':<{W}}\x1b[0m".encode()
        )

        # Instructions
        _draw_text(
            buf,
            3,
            2,
            "Use arrows, or enter number and press Enter, or \x1b[33m[Esc]\x1b[0m to cancel",
        )
        _draw_text(buf, 4, 2, f"Input: \x1b[1;97m{input_buf}\x1b[5m_\x1b[0m")

        # Grid
        start_row = 6
        for row_i in range(min(visible_rows, rows - scroll_offset)):
            actual_row = row_i + scroll_offset
            y = start_row + row_i
            for col_i in range(cols):
                idx = actual_row * cols + col_i
                if idx >= len(items):
                    break
                _, display = items[idx]
                is_cur = idx == current_display_idx
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
            _draw_text(
                buf,
                start_row + visible_rows,
                2,
                f"\x1b[90m({scroll_offset + 1}-{min(scroll_offset + visible_rows, rows)}"
                f" of {rows} rows, arrows move selection)\x1b[0m",
            )

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
            else:
                return items[current_display_idx][0]
            input_buf = ""  # Clear on invalid
        elif key == "\x7f" or key == "\x08":  # Backspace
            input_buf = input_buf[:-1]
        elif key.isdigit():
            input_buf += key
        elif key == "UP":
            input_buf = ""
            current_display_idx = max(0, current_display_idx - cols)
            _clamp_visible_row()
        elif key == "DOWN":
            input_buf = ""
            current_display_idx = min(len(items) - 1, current_display_idx + cols)
            _clamp_visible_row()
        elif key == "LEFT":
            input_buf = ""
            current_display_idx = max(0, current_display_idx - 1)
            _clamp_visible_row()
        elif key == "RIGHT":
            input_buf = ""
            current_display_idx = min(len(items) - 1, current_display_idx + 1)
            _clamp_visible_row()
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
        W, _ = term.cols, term.rows
        buf = io.BytesIO()
        buf.write(b"\x1b[2J\x1b[H")
        buf.write(
            f"\x1b[1;1H\x1b[48;5;24m\x1b[97;1m{f'  {title}  ':{W}}\x1b[0m".encode()
        )
        if subtitle:
            _draw_text(buf, 3, 2, subtitle)
        _draw_text(
            buf,
            5,
            2,
            "Use ↑/↓, or enter number and press Enter, or "
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
        if key == "UP" or key == "LEFT":
            input_buf = ""
            current_idx = max(0, current_idx - 1)
            continue
        if key == "DOWN" or key == "RIGHT":
            input_buf = ""
            current_idx = min(len(options) - 1, current_idx + 1)
            continue
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


def _screen_cameras(
    term: _Term, out, devices: list[DetectedDevice], total_steps: int = 5
) -> list[CameraConfig] | None:
    """Camera identification screen — live preview + name prompt."""
    configs: list[CameraConfig] = []

    mode_idx = 0  # start at "true" (24-bit truecolor)
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
        phase = "preview"  # "preview" → "name" → accepted
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
            _draw_text(
                buf,
                3,
                2,
                f"Camera {i+1}/{len(devices)}: "
                f"\x1b[96m{dev.label}\x1b[0m  "
                f"\x1b[90m[{MODE_LABELS[render_mode]}]\x1b[0m",
            )
            _draw_text(buf, 4, 2, f"Type: {dev.dtype}  Device: {dev.device_id}")
            settings_row = 5
            if dev.id_path:
                # Truncate id_path if too long
                id_path_display = (
                    dev.id_path
                    if len(dev.id_path) <= W - 14
                    else dev.id_path[: W - 17] + "..."
                )
                _draw_text(
                    buf, settings_row, 2, f"\x1b[90mID_PATH: {id_path_display}\x1b[0m"
                )
                settings_row += 1

            # Format/mode selection (for v4l2)
            if formats and dev.dtype == "v4l2":
                cur_fmt = formats[format_idx] if format_idx < len(formats) else None
                modes = cur_fmt.modes if cur_fmt else []

                # Format selector
                fmt_str = cur_fmt.fourcc if cur_fmt else "N/A"
                fmt_desc = cur_fmt.description if cur_fmt else ""
                _draw_text(
                    buf,
                    settings_row,
                    2,
                    f"\x1b[33m[f]\x1b[0m Format: "
                    f"\x1b[1;97m{fmt_str}\x1b[0m "
                    f"\x1b[90m({fmt_desc})\x1b[0m",
                )

                # Mode selector
                if modes and mode_sel_idx < len(modes):
                    cur_mode = modes[mode_sel_idx]
                    mode_str = f"{cur_mode.width}×{cur_mode.height}@{cur_mode.fps}fps"
                else:
                    mode_str = f"{cur_width}×{cur_height}@{cur_fps}fps"
                _draw_text(
                    buf,
                    settings_row + 1,
                    2,
                    f"\x1b[33m[r]\x1b[0m Resolution: "
                    f"\x1b[1;97m{mode_str}\x1b[0m "
                    f"\x1b[90m({len(modes)} available)\x1b[0m",
                )
                settings_row += 2

            # Depth/IR visualization mode display
            if is_depth_camera:
                depth_mode_str = DEPTH_MODES[depth_mode_idx]
                _draw_text(
                    buf,
                    settings_row,
                    2,
                    f"\x1b[33m[d]\x1b[0m Visualization: "
                    f"\x1b[1;97m{MODE_LABELS.get(depth_mode_str, depth_mode_str)}\x1b[0m",
                )
                settings_row += 1

            # Format/resolution for RealSense channels
            if formats and dev.dtype.startswith("realsense_"):
                cur_fmt = formats[format_idx] if format_idx < len(formats) else None
                modes = cur_fmt.modes if cur_fmt else []

                # Format selector
                fmt_str = cur_fmt.fourcc if cur_fmt else "N/A"
                fmt_desc = cur_fmt.description if cur_fmt else ""
                _draw_text(
                    buf,
                    settings_row,
                    2,
                    f"\x1b[33m[f]\x1b[0m Format: "
                    f"\x1b[1;97m{fmt_str}\x1b[0m "
                    f"\x1b[90m({fmt_desc})\x1b[0m",
                )

                # Mode selector
                mode_str = f"{cur_width}×{cur_height}@{cur_fps}fps"
                _draw_text(
                    buf,
                    settings_row + 1,
                    2,
                    f"\x1b[33m[r]\x1b[0m Resolution: "
                    f"\x1b[1;97m{mode_str}\x1b[0m "
                    f"\x1b[90m({len(modes)} available)\x1b[0m",
                )
                settings_row += 2

            # Live preview
            preview_y = settings_row + 1
            avail_h = max(4, H - preview_y - 5)
            avail_w = W - 4
            preview_frame = None

            if cam is not None:
                _, preview_frame = cam.read()
                if preview_frame is not None:
                    frame_h, frame_w = preview_frame.shape[:2]
                    preview_w, preview_h = calc_render_size(
                        frame_w, frame_h, avail_w, avail_h
                    )
                    # Use appropriate renderer based on device type
                    if is_depth_camera and preview_frame.ndim == 2:
                        # Grayscale/depth frame - use depth renderer
                        depth_mode = DEPTH_MODES[depth_mode_idx]
                        rendered = render_depth(
                            preview_frame, preview_w, preview_h, depth_mode
                        )
                    else:
                        # Color frame - use standard renderer
                        rendered = render_frame(
                            preview_frame, preview_w, preview_h, render_mode
                        )
                    buf.write(blit_frame(rendered, preview_y, 3))
                else:
                    preview_w, preview_h = calc_render_size(
                        cur_width, cur_height, avail_w, avail_h
                    )
            else:
                preview_w, preview_h = calc_render_size(
                    cur_width, cur_height, avail_w, avail_h
                )
                _draw_text(buf, preview_y, 3, "(no preview available)")

            prompt_row = preview_y + preview_h + 1

            if phase == "preview":
                # Live-preview phase
                controls = (
                    "\x1b[33m[Enter]\x1b[0m name  "
                    "\x1b[33m[s]\x1b[0m skip  "
                    "\x1b[33m[m]\x1b[0m color  "
                )
                if formats and dev.dtype == "v4l2":
                    controls += "\x1b[33m[f]\x1b[0m format  " "\x1b[33m[r]\x1b[0m res  "
                if formats and dev.dtype.startswith("realsense_"):
                    controls += "\x1b[33m[f]\x1b[0m format  " "\x1b[33m[r]\x1b[0m res  "
                if is_depth_camera:
                    controls += "\x1b[33m[d]\x1b[0m viz-mode  "
                controls += "\x1b[33m[\\]\x1b[0m debug  " "\x1b[33m[Esc/q]\x1b[0m quit"
                _draw_text(buf, prompt_row, 2, controls)

                # FPS tracking
                _t_now = time.monotonic()
                _dt = _t_now - _t_prev
                _t_prev = _t_now
                if _dt > 0:
                    _fps = 0.9 * _fps + 0.1 / _dt

                if show_debug:
                    _draw_text(
                        buf,
                        2,
                        W - 18,
                        f"\x1b[48;5;234m\x1b[38;5;82m FPS: {_fps:5.1f} \x1b[0m",
                    )

                # Erase leftover content below prompt
                buf.write(f"\x1b[{prompt_row + 1};1H\x1b[J".encode())

                _clear = b"\x1b[2J" if _needs_clear else b""
                _needs_clear = False
                out.write(_SY_S + b"\x1b[H" + _clear + buf.getvalue() + _SY_E)
                out.flush()

                key = term.read_key()
                if key == "\n" or key == "\r":
                    phase = "name"
                    _needs_clear = True
                elif key == "s":
                    break  # skip this camera
                elif key == "m":
                    mode_idx = (mode_idx + 1) % len(RENDER_MODES)
                elif (
                    key == "f"
                    and formats
                    and (dev.dtype == "v4l2" or dev.dtype.startswith("realsense_"))
                ):
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
                elif (
                    key == "r"
                    and formats
                    and (dev.dtype == "v4l2" or dev.dtype.startswith("realsense_"))
                ):
                    # Open resolution picker
                    cur_fmt = formats[format_idx] if format_idx < len(formats) else None
                    if cur_fmt and cur_fmt.modes:
                        selected = _pick_resolution(
                            term, out, cur_fmt.modes, mode_sel_idx, W, H
                        )
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
                    return None  # cancel wizard early

                # Throttle to ~60 FPS
                _el = time.monotonic() - _t_now
                if _el < 0.016:
                    time.sleep(0.016 - _el)

            elif phase == "name":
                # Name-input phase
                _draw_text(
                    buf,
                    prompt_row + 1,
                    2,
                    "\x1b[33m[Enter]\x1b[0m accept  "
                    "\x1b[33m[Esc]\x1b[0m back to preview",
                )

                out.write(_SY_S + b"\x1b[H\x1b[2J" + buf.getvalue() + _SY_E)
                out.flush()

                result = _prompt_line(
                    term, out, prompt_row, 2, "Channel name: ", default_name
                )

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
                channel = dev.properties.get(
                    "channel", dev.dtype.replace("realsense_", "")
                )

            configs.append(
                CameraConfig(
                    name=chosen_name,
                    type=cam_type,
                    device=dev.device_id,
                    width=cur_width,
                    height=cur_height,
                    fps=cur_fps,
                    pixel_format=cur_format,
                    id_path=dev.id_path,
                    channel=channel,
                )
            )

    return configs


def _get_airbot_robot(dev: DetectedDevice):
    """Get a runtime robot instance for AIRBOT identification preview."""
    if not str(dev.dtype).startswith("airbot_"):
        return None
    try:
        from rollio.robot import AIRBOTE2B, AIRBOTG2, AIRBOTPlay, is_airbot_available

        if not is_airbot_available():
            return None
        can_interface = dev.properties.get("can_interface", str(dev.device_id))
        if dev.dtype == "airbot_play" and AIRBOTPlay is not None:
            return AIRBOTPlay(can_interface=can_interface)
        if dev.dtype == "airbot_e2b" and AIRBOTE2B is not None:
            return AIRBOTE2B(can_interface=can_interface)
        if dev.dtype == "airbot_g2" and AIRBOTG2 is not None:
            return AIRBOTG2(can_interface=can_interface)
        return None
    except (OSError, RuntimeError, ValueError, TypeError, ImportError):
        return None


def _screen_robots(
    term: _Term, out, devices: list[DetectedDevice], _total_steps: int = 5
) -> list[RobotConfig] | None:
    """Robot identification screen — oscillation + name prompt."""
    configs: list[RobotConfig] = []
    show_debug = False
    _t_prev = time.monotonic()
    _fps = 0.0

    for i, dev in enumerate(devices):
        rob = _make_robot(dev)
        chosen_name: str | None = None
        chosen_role: str = _default_robot_role(dev.dtype)
        chosen_target_tracking_mode: str = "mit"
        default_name = _default_robot_name(dev.dtype, i)
        phase = "preview"  # "preview" → "name" → "role" → "tracking" → done
        _needs_clear = True

        # Start identification for AIRBOT devices
        airbot_robot = _get_airbot_robot(dev)
        if airbot_robot is not None:
            try:
                airbot_robot.identify_start()
            except (OSError, RuntimeError, ValueError, TypeError):
                pass

        while chosen_name is None or phase != "done":
            W, H = term.cols, term.rows

            buf = io.BytesIO()

            _draw_header(buf, W, 2, 3, "Robots")

            _draw_text(
                buf, 3, 2, f"Robot {i+1}/{len(devices)}: " f"\x1b[96m{dev.label}\x1b[0m"
            )
            _draw_text(
                buf,
                4,
                2,
                f"Type: {dev.dtype}  Joints: "
                f"{dev.properties.get('num_joints', '?')}",
            )

            led_block = _airbot_led_block()
            if dev.dtype == "pseudo":
                _draw_text(
                    buf,
                    5,
                    2,
                    "\x1b[93m⟳ Last joint oscillating "
                    "(simulated identification)\x1b[0m",
                )
            elif dev.dtype == "airbot_play" and airbot_robot is not None:
                _draw_text(
                    buf,
                    5,
                    2,
                    f"\x1b[93m⟳ LED blinking orange {led_block} + gravity compensation "
                    "(move arm to identify)\x1b[0m",
                )
            elif dev.dtype == "airbot_e2b" and airbot_robot is not None:
                _draw_text(
                    buf,
                    5,
                    2,
                    f"\x1b[93m⟳ LED blinking orange {led_block} + live E2B position "
                    "(move E2B to identify)\x1b[0m",
                )
            elif dev.dtype == "airbot_g2" and airbot_robot is not None:
                _draw_text(
                    buf,
                    5,
                    2,
                    f"\x1b[93m⟳ LED blinking orange {led_block} + G2 oscillation "
                    "(0-70mm identify motion)\x1b[0m",
                )

            # Robot state display (re-read every iteration)
            pos = None
            n_joints = dev.properties.get("num_joints", 6)
            command_debug: tuple[str, str] | None = None

            # Try AIRBOT robot first (for real hardware)
            if (
                str(dev.dtype).startswith("airbot_")
                and airbot_robot is not None
                and airbot_robot._is_open
            ):
                try:
                    airbot_robot.identify_step()
                    joint_state = airbot_robot.read_joint_state()
                    if joint_state.is_valid and joint_state.position is not None:
                        pos = joint_state.position
                        n_joints = len(pos)
                    debug_getter = getattr(airbot_robot, "latest_command_debug", None)
                    if callable(debug_getter):
                        command_debug = debug_getter()  # pylint: disable=not-callable
                except (OSError, RuntimeError, ValueError, TypeError):
                    pass

            # Fall back to legacy sensor interface (for pseudo robot)
            if pos is None and rob is not None:
                _, state = rob.read()
                pos = state.get("position", np.zeros(0))
                n_joints = len(pos) if len(pos) > 0 else n_joints

            # Display joint positions
            start_row = 7
            state_rows = 0
            if pos is not None and len(pos) > 0:
                bar_w = min(40, W - 25)
                for j in range(n_joints):
                    p = pos[j] if j < len(pos) else 0.0
                    value_text, frac = _format_joint_preview(dev.dtype, float(p))
                    bar_len = int(frac * bar_w)
                    bar = "█" * bar_len + "░" * (bar_w - bar_len)
                    # For pseudo robot, mark last joint as moving
                    marker = ""
                    if dev.dtype == "pseudo" and j == n_joints - 1:
                        marker = " \x1b[91m← moving\x1b[0m"
                    _draw_text(
                        buf,
                        start_row + j,
                        4,
                        f"j{j} \x1b[36m{value_text}\x1b[0m "
                        f"\x1b[33m{bar}\x1b[0m{marker}",
                    )
                state_rows = n_joints

            command_rows = 0
            if dev.dtype in {"airbot_e2b", "airbot_g2"} and command_debug is not None:
                cmd_type, args_text = command_debug
                max_text_w = max(16, W - 8)
                cmd_line = f"Cmd: {cmd_type}"
                args_line = f"Args: {args_text}"
                if len(cmd_line) > max_text_w:
                    cmd_line = cmd_line[: max_text_w - 3] + "..."
                if len(args_line) > max_text_w:
                    args_line = args_line[: max_text_w - 3] + "..."
                cmd_row = start_row + max(state_rows, 1) + 1
                _draw_text(buf, cmd_row, 4, f"\x1b[90m{cmd_line}\x1b[K")
                _draw_text(buf, cmd_row + 1, 4, f"\x1b[90m{args_line}\x1b[K")
                command_rows = 2

            prompt_row = max(
                start_row
                + max(dev.properties.get("num_joints", 6), state_rows, 1)
                + command_rows
                + 2,
                H - 5,
            )

            if phase == "preview":
                # Live-preview phase: robot state refreshes every iteration,
                # shortcut keys are handled directly.
                _draw_text(
                    buf,
                    prompt_row,
                    2,
                    "\x1b[33m[Enter]\x1b[0m name  "
                    "\x1b[33m[s]\x1b[0m skip  "
                    "\x1b[33m[\\]\x1b[0m debug  "
                    "\x1b[33m[Esc/q]\x1b[0m quit",
                )

                # FPS tracking
                _t_now = time.monotonic()
                _dt = _t_now - _t_prev
                _t_prev = _t_now
                if _dt > 0:
                    _fps = 0.9 * _fps + 0.1 / _dt

                if show_debug:
                    _draw_text(
                        buf,
                        2,
                        W - 18,
                        f"\x1b[48;5;234m\x1b[38;5;82m FPS: {_fps:5.1f} \x1b[0m",
                    )

                buf.write(f"\x1b[{prompt_row + 1};1H\x1b[J".encode())

                _clear = b"\x1b[2J" if _needs_clear else b""
                _needs_clear = False
                out.write(_SY_S + b"\x1b[H" + _clear + buf.getvalue() + _SY_E)
                out.flush()

                key = term.read_key()
                if key == "\n" or key == "\r":
                    phase = "name"
                    _needs_clear = True
                elif key == "s":
                    break  # skip this robot (identify_stop called below)
                elif key == "\\":
                    show_debug = not show_debug
                elif key == "q" or key == "\x1b":
                    # Stop identification before quitting
                    if airbot_robot is not None:
                        try:
                            airbot_robot.identify_stop()
                        except (OSError, RuntimeError, ValueError, TypeError):
                            pass
                    if rob:
                        rob.close()
                    return None

                # Throttle to ~60 FPS
                _el = time.monotonic() - _t_now
                if _el < 0.016:
                    time.sleep(0.016 - _el)

            elif phase == "name":
                _draw_text(
                    buf,
                    prompt_row + 1,
                    2,
                    "\x1b[33m[Enter]\x1b[0m accept  \x1b[33m[Esc]\x1b[0m back",
                )

                out.write(_SY_S + b"\x1b[H\x1b[2J" + buf.getvalue() + _SY_E)
                out.flush()

                result = _prompt_line(
                    term, out, prompt_row, 2, "Channel name: ", default_name
                )
                if result is None:
                    phase = "preview"
                    _needs_clear = True
                    continue
                chosen_name = result
                phase = "role"

            elif phase == "role":
                _draw_text(buf, prompt_row, 2, f"Name: \x1b[97;1m{chosen_name}\x1b[0m")
                _draw_text(
                    buf,
                    prompt_row + 1,
                    2,
                    "\x1b[33m[Enter]\x1b[0m accept  \x1b[33m[Esc]\x1b[0m back",
                )

                out.write(_SY_S + b"\x1b[H\x1b[2J" + buf.getvalue() + _SY_E)
                out.flush()

                result = _prompt_line(
                    term,
                    out,
                    prompt_row - 1,
                    2,
                    "Role ([f]ollower / [l]eader): ",
                    "l" if chosen_role == "leader" else "f",
                )
                if result is None:
                    phase = "name"
                    continue
                chosen_role = "leader" if result.lower().startswith("l") else "follower"
                phase = (
                    "tracking" if dev.dtype in {"airbot_play", "airbot_g2"} else "done"
                )

            elif phase == "tracking":
                _draw_text(
                    buf, prompt_row - 1, 2, f"Name: \x1b[97;1m{chosen_name}\x1b[0m"
                )
                _draw_text(buf, prompt_row, 2, f"Role: \x1b[97;1m{chosen_role}\x1b[0m")
                _draw_text(
                    buf,
                    prompt_row + 1,
                    2,
                    "\x1b[33m[Enter]\x1b[0m accept  \x1b[33m[Esc]\x1b[0m back",
                )

                out.write(_SY_S + b"\x1b[H\x1b[2J" + buf.getvalue() + _SY_E)
                out.flush()

                result = _prompt_line(
                    term,
                    out,
                    prompt_row - 2,
                    2,
                    "Target tracking ([m]it / [p]vt): ",
                    "p" if chosen_target_tracking_mode == "pvt" else "m",
                )
                if result is None:
                    phase = "role"
                    continue
                chosen_target_tracking_mode = (
                    "pvt" if result.lower().startswith("p") else "mit"
                )
                phase = "done"

        # Stop identification for AIRBOT devices
        if airbot_robot is not None:
            try:
                airbot_robot.identify_stop()
            except (OSError, RuntimeError, ValueError, TypeError):
                pass

        if rob:
            rob.close()

        if chosen_name:
            robot_options = {}
            if dev.dtype in {"airbot_play", "airbot_g2"}:
                robot_options["target_tracking_mode"] = chosen_target_tracking_mode
            configs.append(
                RobotConfig(
                    name=chosen_name,
                    type=dev.dtype,
                    role=chosen_role,
                    num_joints=dev.properties.get("num_joints", 6),
                    device=str(dev.properties.get("can_interface", dev.device_id)),
                    options=robot_options,
                )
            )

    return configs


def _screen_settings(
    term: _Term,
    out,
    robots: list[RobotConfig],
    *,
    step: int = 3,
    total_steps: int = 5,
) -> tuple[str, str, str, str, str, bool] | None:
    """Project/settings screen — project, storage, mode, and codecs."""
    W, _ = term.cols, term.rows

    out.write(_SY_S + b"\x1b[2J")
    _draw_header(out, W, step, total_steps, "Project Settings")

    _draw_text(out, 4, 2, "Configure project, collection mode, and codecs.")
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

    plotjuggler_idx = _pick_option(
        term,
        out,
        title="PLOTJUGGLER VISUALIZATION",
        options=["Off", "On"],
        current_idx=0,
        subtitle="Stream robot joint positions to PlotJuggler over UDP.",
    )
    if plotjuggler_idx is None:
        return None
    plotjuggler_enabled = plotjuggler_idx == 1

    return name, storage, mode, rgb_codec, depth_codec, plotjuggler_enabled


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
        _draw_header(
            out,
            W,
            step,
            total_steps,
            f"Tele-op Pairing ({idx + 1}/{min(len(leaders), len(followers))})",
        )
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

        pairs.append(
            TeleopPairConfig(
                name=f"pair_{idx}",
                leader=leader.name,
                follower=follower.name,
                mapper=mapper,
            )
        )

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
                if di not in used_cam_devs and _camera_types_match(
                    dev.dtype, cc.type, cc.channel
                ):
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
            same_type = dev.dtype == rc.type
            same_device = str(
                dev.properties.get("can_interface", dev.device_id)
            ) == str(rc.device)
            if di not in used_rob_devs and same_type and same_device:
                matched_dev = dev
                used_rob_devs.add(di)
                break
        if matched_dev is None:
            for di, dev in enumerate(rob_devs):
                if di not in used_rob_devs and dev.dtype == rc.type:
                    matched_dev = dev
                    used_rob_devs.add(di)
                    break
        if matched_dev is None:
            for di, dev in enumerate(rob_devs):
                same_device = str(
                    dev.properties.get("can_interface", dev.device_id)
                ) == str(rc.device)
                if di not in used_rob_devs and same_device:
                    matched_dev = dev
                    used_rob_devs.add(di)
                    break
        matched.append((rc, matched_dev))
    return matched


def _screen_summary(
    term: _Term,
    out,
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
    plotjuggler_enabled: bool,
    teleop_pairs: list[TeleopPairConfig],
    step: int = 5,
    total_steps: int = 5,
) -> bool:
    """Live summary screen with camera previews + robot states.

    Returns True to save, False to cancel.
    """
    cam_entries = _match_camera_devices(cam_configs, cam_devs)
    rob_entries = _match_robot_devices(rob_configs, rob_devs)

    mode_idx = 0  # "true" (24-bit truecolor) for best preview quality
    show_debug = False
    _t_prev = time.monotonic()
    _fps = 0.0
    _render_loop_count = 0
    _render_last_loop_us = 0.0
    _render_avg_loop_us = 0.0
    _render_gap_history_ms: deque[float] = deque(maxlen=64)
    _render_work_history_ms: deque[float] = deque(maxlen=64)
    result = None
    _needs_clear = True
    selected_cam = 0 if cam_entries else -1
    _needs_restart = True
    preview_runtime: AsyncCollectionRuntime | None = None
    preview_started_at: float | None = None
    preview_target_fps = max([30, *[cfg.fps for cfg in cam_configs]])

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
        preview_cfg = RollioConfig(
            project_name=project_name,
            fps=preview_target_fps,
            mode=mode,
            plotjuggler_enabled=plotjuggler_enabled,
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
            scheduler_driver="round_robin",
            preview_live_feedback=True,
        )

    def _aggregate_task_rate(
        task_metrics: dict[str, object], prefix: str
    ) -> tuple[float | None, int, float | None]:
        if preview_started_at is None:
            return None, 0, None
        runtime_age = max(time.monotonic() - preview_started_at, 1e-6)
        matching = [
            metric for name, metric in task_metrics.items() if name.startswith(prefix)
        ]
        if not matching:
            return None, 0, None
        rate_hz = sum(metric.run_count / runtime_age for metric in matching) / len(
            matching
        )
        overruns = sum(metric.overrun_count for metric in matching)
        avg_step_ms = sum(metric.avg_step_ms for metric in matching) / len(matching)
        return rate_hz, overruns, avg_step_ms

    def _render_rate_text(target_hz: int, observed_hz: float | None) -> str:
        if observed_hz is None:
            return f"{target_hz}Hz target / n/a actual"
        return f"{target_hz}Hz target / {observed_hz:5.1f}Hz actual"

    try:
        while result is None:
            loop_started_at = time.monotonic()
            if _needs_restart:
                if preview_runtime is not None:
                    preview_runtime.close()

                def _start_preview_runtime() -> AsyncCollectionRuntime:
                    runtime = _build_preview_runtime()
                    runtime.open()
                    return runtime

                preview_runtime = _run_with_loading(
                    term,
                    out,
                    step=step,
                    total=total_steps,
                    title="Summary — Live Preview",
                    message="Starting live preview runtime...",
                    work=_start_preview_runtime,
                )
                preview_started_at = time.monotonic()
                _needs_clear = True
                _needs_restart = False

            W, H = term.cols, term.rows
            buf = io.BytesIO()
            latest_frames = (
                preview_runtime.latest_frames() if preview_runtime is not None else {}
            )
            latest_robot_states = (
                preview_runtime.latest_robot_states()
                if preview_runtime is not None
                else {}
            )
            driver_metrics = (
                preview_runtime.scheduler_metrics()["driver"]
                if preview_runtime is not None
                else None
            )
            diagnostics_getter = getattr(preview_runtime, "timing_diagnostics", None)
            timing_diagnostics = (
                diagnostics_getter()
                if preview_runtime is not None and callable(diagnostics_getter)
                else None
            )
            task_metrics = (
                driver_metrics.task_metrics if driver_metrics is not None else {}
            )
            telemetry_actual_hz, telemetry_overruns, telemetry_avg_step_ms = (
                _aggregate_task_rate(
                    task_metrics,
                    "robot-",
                )
            )
            control_actual_hz, control_overruns, control_avg_step_ms = (
                _aggregate_task_rate(
                    task_metrics,
                    "teleop-",
                )
            )
            driver_last_loop_us = (
                driver_metrics.last_loop_us
                if driver_metrics is not None and driver_metrics.loop_run_count > 0
                else None
            )
            driver_avg_loop_us = (
                driver_metrics.avg_loop_us
                if driver_metrics is not None and driver_metrics.loop_run_count > 0
                else None
            )

            # FPS tracking
            _t_now = time.monotonic()
            _dt = _t_now - _t_prev
            _t_prev = _t_now
            if _dt > 0:
                _fps = 0.9 * _fps + 0.1 / _dt
                _render_gap_history_ms.append(_dt * 1000.0)

            _draw_header(buf, W, step, total_steps, "Summary — Live Preview")

            # ── Layout: left panel (info), right panel (live previews) ──
            info_w = max(40, W // 3)
            preview_w = W - info_w - 2

            # ── Left panel: configuration info ──
            row = 3
            box_line = "─" * (info_w - 4)
            _draw_text_clear(buf, row, 2, "┌" + box_line + "┐", info_w)
            row += 1
            _draw_text_clear(
                buf,
                row,
                2,
                f"│ \x1b[1mProject:\x1b[0m \x1b[97;1m{project_name[:info_w-14]}\x1b[0m",
                info_w,
            )
            row += 1
            _draw_text_clear(
                buf,
                row,
                2,
                f"│ \x1b[1mStorage:\x1b[0m \x1b[90m{storage_root[:info_w-14]}\x1b[0m",
                info_w,
            )
            row += 1
            _draw_text_clear(
                buf, row, 2, f"│ \x1b[1mMode:\x1b[0m \x1b[97;1m{mode}\x1b[0m", info_w
            )
            row += 1
            _draw_text_clear(
                buf,
                row,
                2,
                f"│ \x1b[1mRGB codec:\x1b[0m \x1b[90m{video_codec[:info_w-16]}\x1b[0m",
                info_w,
            )
            row += 1
            _draw_text_clear(
                buf,
                row,
                2,
                f"│ \x1b[1mDepth codec:\x1b[0m \x1b[90m{depth_codec[:info_w-18]}\x1b[0m",
                info_w,
            )
            row += 1
            pj_state = "\x1b[92mon\x1b[0m" if plotjuggler_enabled else "\x1b[90moff\x1b[0m"
            _draw_text_clear(
                buf,
                row,
                2,
                f"│ \x1b[1mPlotJuggler:\x1b[0m {pj_state}",
                info_w,
            )
            row += 1
            if preview_runtime is not None:
                _draw_text_clear(
                    buf,
                    row,
                    2,
                    f"│ \x1b[1mDriver:\x1b[0m \x1b[90m{preview_runtime.scheduler_driver}\x1b[0m",
                    info_w,
                )
                row += 1
                _draw_text_clear(
                    buf,
                    row,
                    2,
                    f"│ \x1b[1mTelemetry:\x1b[0m \x1b[90m{_render_rate_text(preview_runtime.telemetry_hz, telemetry_actual_hz)}\x1b[0m",
                    info_w,
                )
                row += 1
                _draw_text_clear(
                    buf,
                    row,
                    2,
                    f"│ \x1b[1mControl:\x1b[0m \x1b[90m{_render_rate_text(preview_runtime.control_hz, control_actual_hz)}\x1b[0m",
                    info_w,
                )
                row += 1
                _draw_text_clear(
                    buf,
                    row,
                    2,
                    f"│ \x1b[1mOverruns:\x1b[0m \x1b[90mtelem {telemetry_overruns} / ctrl {control_overruns}\x1b[0m",
                    info_w,
                )
                row += 1
                step_parts = []
                if telemetry_avg_step_ms is not None:
                    step_parts.append(f"telem {telemetry_avg_step_ms:4.1f}ms")
                if control_avg_step_ms is not None:
                    step_parts.append(f"ctrl {control_avg_step_ms:4.1f}ms")
                if step_parts:
                    _draw_text_clear(
                        buf,
                        row,
                        2,
                        f"│ \x1b[1mAvg step:\x1b[0m \x1b[90m{' / '.join(step_parts)}\x1b[0m",
                        info_w,
                    )
                    row += 1
                if driver_last_loop_us is not None and driver_avg_loop_us is not None:
                    _draw_text_clear(
                        buf,
                        row,
                        2,
                        f"│ \x1b[1mMain loop:\x1b[0m \x1b[90m{driver_last_loop_us:6.0f}us last / {driver_avg_loop_us:6.0f}us avg\x1b[0m",
                        info_w,
                    )
                    row += 1
                if _render_loop_count > 0:
                    _draw_text_clear(
                        buf,
                        row,
                        2,
                        f"│ \x1b[1mRender:\x1b[0m \x1b[90m{_render_last_loop_us:6.0f}us last / {_render_avg_loop_us:6.0f}us avg\x1b[0m",
                        info_w,
                    )
                    row += 1
                if show_debug:
                    target_render_ms = 1000.0 / max(preview_target_fps, 1)
                    debug_panel_h = max(8, min(16, H - row - 8))
                    debug_lines = build_timing_panel_lines(
                        panel_w=max(info_w - 4, 16),
                        panel_h=debug_panel_h,
                        diagnostics=timing_diagnostics,
                        render_gap_trace=make_timing_trace(
                            tuple(_render_gap_history_ms),
                            target_interval_ms=target_render_ms,
                            age_ms=0.0 if _render_gap_history_ms else None,
                        ),
                        render_work_trace=make_timing_trace(
                            tuple(_render_work_history_ms),
                            target_interval_ms=target_render_ms,
                        ),
                    )
                    for line in debug_lines:
                        _draw_text_clear(buf, row, 2, f"│ {line}", info_w)
                        row += 1
            _draw_text_clear(buf, row, 2, "├" + box_line + "┤", info_w)
            row += 1
            _draw_text_clear(
                buf,
                row,
                2,
                (
                    f"│ \x1b[1;96mCameras ({len(cam_configs)})\x1b[0m "
                    f"\x1b[90m[1-{len(cam_configs)}] select\x1b[0m"
                    if cam_configs
                    else "│ \x1b[1;96mCameras (0)\x1b[0m"
                ),
                info_w,
            )
            row += 1
            for ci, (cc, _) in enumerate(cam_entries):
                is_sel = ci == selected_cam
                sel_mark = "\x1b[97;1m▸\x1b[0m" if is_sel else " "
                highlight = "\x1b[97;1;44m" if is_sel else "\x1b[96m"
                # Line 1: index and name
                _draw_text_clear(
                    buf,
                    row,
                    2,
                    f"│ {sel_mark}{ci+1}.{highlight}{cc.name[:12]}\x1b[0m",
                    info_w,
                )
                row += 1
                # Line 2: type | format | resolution | fps (aligned)
                type_str = cc.type
                if cc.type == "realsense":
                    # Abbreviate channel names: color→col, depth→dep, infrared→ir
                    ch_abbr = {"color": "col", "depth": "dep", "infrared": "ir"}.get(
                        cc.channel, cc.channel or "col"
                    )
                    type_str = f"rs:{ch_abbr}"
                fmt_str = cc.pixel_format or "?"
                res_str = f"{cc.width}×{cc.height}"
                fps_str = f"{cc.fps}fps" if cc.fps else "?fps"
                _draw_text_clear(
                    buf,
                    row,
                    2,
                    f"│    \x1b[90m{type_str:<8} {fmt_str:<5} {res_str:<10} {fps_str}\x1b[0m",
                    info_w,
                )
                row += 1
            _draw_text_clear(buf, row, 2, "├" + box_line + "┤", info_w)
            row += 1
            _draw_text_clear(
                buf, row, 2, f"│ \x1b[1;93mRobots ({len(rob_configs)})\x1b[0m", info_w
            )
            row += 1
            for rc, matched_rob_dev in rob_entries:
                role_clr = "92" if rc.role == "leader" else "33"
                _draw_text_clear(
                    buf,
                    row,
                    2,
                    f"│  \x1b[93m{rc.name[:12]:<12}\x1b[0m "
                    f"\x1b[{role_clr}m{rc.role}\x1b[0m "
                    f"\x1b[90m{rc.num_joints}-DOF\x1b[0m",
                    info_w,
                )
                row += 1
                if rc.type.startswith("airbot_"):
                    sn = ""
                    eef = ""
                    if matched_rob_dev is not None:
                        sn = matched_rob_dev.properties.get("serial_number", "")
                        eef = matched_rob_dev.properties.get("end_effector_type", "")
                    info_parts: list[str] = []
                    if sn:
                        info_parts.append(f"SN:{sn}")
                    if eef:
                        info_parts.append(f"EEF:{eef}")
                    if rc.type in {"airbot_play", "airbot_g2"}:
                        tracking_mode = (
                            str(rc.options.get("target_tracking_mode", "mit"))
                            .strip()
                            .upper()
                        )
                        info_parts.append(f"TRACK:{tracking_mode}")
                    info_line = " ".join(info_parts)
                    if info_line:
                        _draw_text_clear(
                            buf, row, 2, f"│     \x1b[90m{info_line}\x1b[0m", info_w
                        )
                        row += 1
            if teleop_pairs:
                _draw_text_clear(buf, row, 2, "├" + box_line + "┤", info_w)
                row += 1
                _draw_text_clear(
                    buf,
                    row,
                    2,
                    f"│ \x1b[1;95mTele-op pairs ({len(teleop_pairs)})\x1b[0m",
                    info_w,
                )
                row += 1
                for pair in teleop_pairs:
                    mapper_label = (
                        "direct" if pair.mapper == "joint_direct" else "FK-IK"
                    )
                    pair_text = f"│  \x1b[95m{pair.leader}\x1b[0m → \x1b[93m{pair.follower}\x1b[0m"
                    _draw_text_clear(buf, row, 2, pair_text, info_w)
                    row += 1
                    _draw_text_clear(
                        buf, row, 2, f"│     \x1b[90m{mapper_label}\x1b[0m", info_w
                    )
                    row += 1
            _draw_text_clear(buf, row, 2, "├" + box_line + "┤", info_w)
            row += 1
            _draw_text_clear(buf, row, 2, "│ \x1b[1mSave to:\x1b[0m", info_w)
            row += 1
            path_display = output_path[: info_w - 6]
            _draw_text_clear(buf, row, 2, f"│  \x1b[90m{path_display}\x1b[0m", info_w)
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
            _ = avail_h - cam_h_total if n_robs else 0  # rob_h for future robot layout

            # Draw camera previews in a grid
            if n_cams > 0:
                mode = RENDER_MODES[mode_idx]
                cam_title = "─── CAMERAS "
                cam_title += "─" * max(0, preview_w - len(cam_title))
                _draw_text(
                    buf,
                    preview_row,
                    preview_col,
                    f"\x1b[1;96m{cam_title[:preview_w]}\x1b[0m",
                )
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

                    is_sel = ci == selected_cam
                    frame = latest_frames.get(cc.name)
                    if frame is not None:
                        try:
                            fh, fw = frame.shape[:2]
                            rw, rh = calc_render_size(fw, fh, cell_w - 2, preview_h_per)
                            if cc.channel in ("depth", "infrared") and frame.ndim == 2:
                                rendered = render_depth(frame, rw, rh, "turbo")
                            else:
                                rendered = render_frame(frame, rw, rh, mode)
                            buf.write(blit_frame(rendered, cell_row, cell_col + 1))
                        except (OSError, RuntimeError, ValueError, TypeError):
                            _draw_text(
                                buf, cell_row, cell_col + 1, "\x1b[90m(err)\x1b[0m"
                            )
                    else:
                        _draw_text(
                            buf, cell_row, cell_col + 1, "\x1b[90m(no preview)\x1b[0m"
                        )

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
                _draw_text(
                    buf,
                    preview_row,
                    preview_col,
                    f"\x1b[1;93m{rob_title[:preview_w]}\x1b[0m",
                )
                preview_row += 1
                bar_w = min(30, preview_w - 20)

                for rc, _ in rob_entries:
                    role_clr = "92" if rc.role == "leader" else "33"
                    _draw_text_clear(
                        buf,
                        preview_row,
                        preview_col,
                        f"\x1b[{role_clr};1m{rc.name}\x1b[0m "
                        f"\x1b[90m({rc.num_joints}-DOF, {rc.role})\x1b[0m",
                        preview_w,
                    )
                    try:
                        state = latest_robot_states.get(rc.name, {})
                        pos = state.get("position")
                        state_lines: list[str] = []
                        if pos is None or len(pos) == 0:
                            state_lines.append("\x1b[90m(no data)\x1b[0m")
                        else:
                            for j, p in enumerate(pos):
                                value_text, frac = _format_joint_preview(
                                    rc.type, float(p)
                                )
                                bar_len = int(frac * bar_w)
                                bar = "█" * bar_len + "░" * (bar_w - bar_len)
                                state_lines.append(
                                    f"j{j} \x1b[36m{value_text}\x1b[0m "
                                    f"\x1b[33m{bar}\x1b[0m"
                                )
                        reserved_rows = max(len(state_lines), 1, rc.num_joints)
                        for j in range(reserved_rows):
                            line = state_lines[j] if j < len(state_lines) else ""
                            _draw_text_clear(
                                buf,
                                preview_row + 1 + j,
                                preview_col + 2,
                                line,
                                max(preview_w - 2, 0),
                            )
                        preview_row += 1 + reserved_rows + 1
                    except (OSError, RuntimeError, ValueError, TypeError):
                        _draw_text_clear(
                            buf,
                            preview_row + 1,
                            preview_col + 2,
                            "\x1b[90m(error)\x1b[0m",
                            max(preview_w - 2, 0),
                        )
                        preview_row += 1 + max(1, rc.num_joints) + 1

            # Debug overlay
            if show_debug:
                _draw_text(
                    buf,
                    2,
                    W - 18,
                    f"\x1b[48;5;234m\x1b[38;5;82m FPS: {_fps:5.1f} \x1b[0m",
                )

            # ── Footer / controls ──
            footer_row = H - 2
            cam_controls = ""
            if selected_cam >= 0 and cam_entries[selected_cam][1]:
                dev = cam_entries[selected_cam][1]
                if (
                    dev
                    and dev.formats
                    and (dev.dtype == "v4l2" or dev.dtype.startswith("realsense_"))
                ):
                    cam_controls = (
                        "\x1b[33m[f]\x1b[0m format  \x1b[33m[r]\x1b[0m resolution  "
                    )
            _draw_text_clear(
                buf,
                footer_row,
                2,
                f"\x1b[33m[Enter]\x1b[0m save  "
                f"{cam_controls}"
                "\x1b[33m[m]\x1b[0m mode  "
                "\x1b[33m[\\]\x1b[0m debug  "
                "\x1b[33m[q/Esc]\x1b[0m cancel",
                0,
            )

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
                if (
                    dev
                    and dev.formats
                    and (dev.dtype == "v4l2" or dev.dtype.startswith("realsense_"))
                ):
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
                if (
                    dev
                    and dev.formats
                    and (dev.dtype == "v4l2" or dev.dtype.startswith("realsense_"))
                ):
                    # Find current format and mode index
                    cur_fmt = None
                    for fmt in dev.formats:
                        if fmt.fourcc == cc.pixel_format:
                            cur_fmt = fmt
                            break
                    if cur_fmt and cur_fmt.modes:
                        cur_mode_idx = 0
                        for mi, m in enumerate(cur_fmt.modes):
                            if (
                                m.width == cc.width
                                and m.height == cc.height
                                and m.fps == cc.fps
                            ):
                                cur_mode_idx = mi
                                break
                        new_idx = _pick_resolution(
                            term, out, cur_fmt.modes, cur_mode_idx, W, H
                        )
                        if new_idx is not None:
                            m = cur_fmt.modes[new_idx]
                            cc.width, cc.height, cc.fps = m.width, m.height, m.fps
                            _needs_restart = True
                        _needs_clear = True

            render_elapsed_us = (time.monotonic() - loop_started_at) * 1_000_000.0
            _render_loop_count += 1
            _render_last_loop_us = render_elapsed_us
            _render_work_history_ms.append(render_elapsed_us / 1000.0)
            if _render_loop_count == 1:
                _render_avg_loop_us = render_elapsed_us
            else:
                prev = _render_loop_count - 1
                _render_avg_loop_us = (
                    (_render_avg_loop_us * prev) + render_elapsed_us
                ) / _render_loop_count

            # Throttle to ~30 FPS
            _el = time.monotonic() - _t_now
            if _el < 0.033:
                time.sleep(0.033 - _el)

    finally:
        if preview_runtime is not None:
            _run_with_loading(
                term,
                out,
                step=step,
                total=total_steps,
                title="Summary — Returning Robots",
                message="Returning robots to zero position...",
                work=lambda: preview_runtime.return_robots_to_zero(timeout=5.0),
            )
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
        if cam_configs is None:
            return None

        _show_loading_transition(
            term,
            out,
            step=2,
            total=total_steps,
            title="Loading Robots",
            message="Preparing robot setup...",
        )
        # Step 2: Robots
        rob_configs = _screen_robots(term, out, rob_devs, _total_steps=total_steps)
        if rob_configs is None:
            return None

        _show_loading_transition(
            term,
            out,
            step=3,
            total=total_steps,
            title="Loading Project Settings",
            message="Preparing project settings...",
        )
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
        (
            project_name,
            storage_root,
            mode,
            video_codec,
            depth_codec,
            plotjuggler_enabled,
        ) = settings

        teleop_pairs: list[TeleopPairConfig] = []
        if mode == "teleop":
            _show_loading_transition(
                term,
                out,
                step=4,
                total=total_steps,
                title="Loading Tele-op Pairing",
                message="Preparing tele-op pairing...",
            )
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
        _show_loading_transition(
            term,
            out,
            step=5,
            total=total_steps,
            title="Loading Live Preview",
            message="Preparing final live preview...",
        )
        should_save = _screen_summary(
            term,
            out,
            cam_configs,
            rob_configs,
            cam_devs,
            rob_devs,
            project_name,
            storage_root,
            output_path,
            mode=mode,
            video_codec=video_codec,
            depth_codec=depth_codec,
            plotjuggler_enabled=plotjuggler_enabled,
            teleop_pairs=teleop_pairs,
            step=5,
            total_steps=total_steps,
        )

        if not should_save:
            return None

    return RollioConfig(
        project_name=project_name,
        mode=mode,
        plotjuggler_enabled=plotjuggler_enabled,
        cameras=cam_configs,
        robots=rob_configs,
        teleop_pairs=teleop_pairs,
        storage=StorageConfig(root=storage_root),
        encoder=EncoderConfig(
            video_codec=video_codec,
            depth_codec=depth_codec,
        ),
    )
