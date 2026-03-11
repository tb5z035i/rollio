#!/usr/bin/env python3
"""TUI Webcam — Rich-based demo with multiple render modes.

Demonstrates live camera preview using the Rich library for layout and
screen management, with the same half-block ANSI rendering used by the
rollio data-collection TUI.

Dependencies::

    pip install opencv-python-headless numpy rich

Usage::

    python tui_webcam_rich.py                # default camera
    python tui_webcam_rich.py --device 1     # camera index 1
    python tui_webcam_rich.py --mode 16      # start in 16-colour mode
    python tui_webcam_rich.py --max-width 80 # cap render width

Controls:
    q           quit
    m           cycle render mode  (256 → 16 → gray → 2 → …)
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.measure import Measurement
from rich.text import Text

# ═══════════════════════════════════════════════════════════════════════
#  Render engine  (self-contained — same algorithm as rollio renderer)
# ═══════════════════════════════════════════════════════════════════════

_D3 = np.empty((256, 3), dtype=np.uint8)
for _i in range(256):
    _D3[_i] = list(f"{_i:03d}".encode())

_C256_TMPL = np.frombuffer(
    b"\x1b[38;5;000;48;5;000m\xe2\x96\x80", dtype=np.uint8
).copy()
_C256_LEN = 23

_16_TMPL = np.frombuffer(b"\x1b[000;000m\xe2\x96\x80", dtype=np.uint8).copy()
_16_LEN = 13

_CUBE_V = np.array([0, 95, 135, 175, 215, 255], dtype=np.int32)
_CUBE_T = np.array([0, 48, 115, 155, 195, 235], dtype=np.int32)

_ANSI16_RGB = np.array(
    [
        [0, 0, 0],
        [170, 0, 0],
        [0, 170, 0],
        [170, 170, 0],
        [0, 0, 170],
        [170, 0, 170],
        [0, 170, 170],
        [170, 170, 170],
        [85, 85, 85],
        [255, 85, 85],
        [85, 255, 85],
        [255, 255, 85],
        [85, 85, 255],
        [255, 85, 255],
        [85, 255, 255],
        [255, 255, 255],
    ],
    dtype=np.int32,
)

_FG16 = np.empty((16, 3), dtype=np.uint8)
for _i, _c in enumerate(
    [30, 31, 32, 33, 34, 35, 36, 37, 90, 91, 92, 93, 94, 95, 96, 97]
):
    _FG16[_i] = list(f"{_c:03d}".encode())

_BG16 = np.empty((16, 3), dtype=np.uint8)
for _i, _c in enumerate(
    [40, 41, 42, 43, 44, 45, 46, 47, 100, 101, 102, 103, 104, 105, 106, 107]
):
    _BG16[_i] = list(f"{_c:03d}".encode())

_2BIT_CHARS = np.array(
    [
        [0xE2, 0xA0, 0x80],
        [0xE2, 0x96, 0x80],
        [0xE2, 0x96, 0x84],
        [0xE2, 0x96, 0x88],
    ],
    dtype=np.uint8,
)

_RST_NL = b"\x1b[0m\n"
_RST = b"\x1b[0m"

MODES = ("256", "16", "gray", "2")
_LABELS = {"256": "256-clr", "16": "16-clr", "gray": "gray", "2": "2-clr"}
_BPP = {"256": 23, "16": 13, "gray": 23, "2": 3}


def _rgb_to_256(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., c].astype(np.int32) for c in range(3))
    ri = np.clip(np.searchsorted(_CUBE_T, r, side="right") - 1, 0, 5)
    gi = np.clip(np.searchsorted(_CUBE_T, g, side="right") - 1, 0, 5)
    bi = np.clip(np.searchsorted(_CUBE_T, b, side="right") - 1, 0, 5)
    ci = 16 + 36 * ri + 6 * gi + bi
    cd = (r - _CUBE_V[ri]) ** 2 + (g - _CUBE_V[gi]) ** 2 + (b - _CUBE_V[bi]) ** 2
    avg = (r + g + b) // 3
    gri = np.clip((avg - 8 + 5) // 10, 0, 23)
    gv = 8 + gri * 10
    gd = (r - gv) ** 2 + (g - gv) ** 2 + (b - gv) ** 2
    return np.where(gd < cd, 232 + gri, ci).astype(np.uint8)


def _nearest_16(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float64)
    ref = _ANSI16_RGB.astype(np.float64)
    ff = (flat**2).sum(axis=1)
    rr = (ref**2).sum(axis=1)
    dist = ff[:, None] + rr[None, :] - 2.0 * flat @ ref.T
    return dist.argmin(axis=1).astype(np.uint8).reshape(rgb.shape[:2])


def _render(bgr: np.ndarray, w: int, h: int, mode: str) -> bytes:
    """Render *bgr* to *w* x *h* half-block ANSI bytes in *mode*."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h * 2), interpolation=cv2.INTER_AREA)

    if mode == "256":
        idx = _rgb_to_256(rgb)
        top, bot = idx[0::2], idx[1::2]
        n = w * h
        buf = np.tile(_C256_TMPL, n).reshape(n, _C256_LEN)
        buf[:, 7:10] = _D3[top.ravel()]
        buf[:, 16:19] = _D3[bot.ravel()]
        rows = buf.reshape(h, w * _C256_LEN)
        return _RST_NL.join(rows[y].tobytes() for y in range(h)) + _RST

    elif mode == "16":
        idx = _nearest_16(rgb)
        top, bot = idx[0::2], idx[1::2]
        n = w * h
        buf = np.tile(_16_TMPL, n).reshape(n, _16_LEN)
        buf[:, 2:5] = _FG16[top.ravel()]
        buf[:, 6:9] = _BG16[bot.ravel()]
        rows = buf.reshape(h, w * _16_LEN)
        return _RST_NL.join(rows[y].tobytes() for y in range(h)) + _RST

    elif mode == "gray":
        lum = rgb.mean(axis=2).astype(np.int32)
        idx = (np.clip((lum - 8 + 5) // 10, 0, 23) + 232).astype(np.uint8)
        top, bot = idx[0::2], idx[1::2]
        n = w * h
        buf = np.tile(_C256_TMPL, n).reshape(n, _C256_LEN)
        buf[:, 7:10] = _D3[top.ravel()]
        buf[:, 16:19] = _D3[bot.ravel()]
        rows = buf.reshape(h, w * _C256_LEN)
        return _RST_NL.join(rows[y].tobytes() for y in range(h)) + _RST

    else:  # "2"
        lum = rgb.mean(axis=2)
        bright = (lum > lum.mean()).astype(np.uint8)
        top, bot = bright[0::2], bright[1::2]
        idx = top * 2 + bot
        flat = _2BIT_CHARS[idx.ravel()]
        rows = flat.reshape(h, w * 3)
        prefix = b"\x1b[97;40m"
        return _RST_NL.join(prefix + rows[y].tobytes() for y in range(h)) + _RST


# ═══════════════════════════════════════════════════════════════════════
#  Background camera grabber
# ═══════════════════════════════════════════════════════════════════════


class CameraGrabber:
    """Reads frames in a background thread so the render loop never blocks."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self) -> None:
        while self._running:
            ok, f = self._cap.read()
            if not ok:
                break
            with self._lock:
                self._frame = f

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._running = False
        self._t.join(timeout=2.0)


# ═══════════════════════════════════════════════════════════════════════
#  Rich renderable — wraps pre-rendered ANSI bytes
# ═══════════════════════════════════════════════════════════════════════


class CameraFrame:
    """Rich renderable that displays a half-block ANSI camera frame."""

    def __init__(self) -> None:
        self._text = Text("")
        self._width = 1

    def update(self, bgr: np.ndarray, w: int, h: int, mode: str) -> int:
        """Re-render.  Returns the raw byte count (before Rich re-encoding)."""
        raw = _render(bgr, w, h, mode)
        self._text = Text.from_ansi(raw.decode("utf-8", errors="replace"))
        self._width = w
        return len(raw)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield self._text

    def __rich_measure__(
        self, console: Console, options: ConsoleOptions
    ) -> Measurement:
        return Measurement(self._width, self._width)


# ═══════════════════════════════════════════════════════════════════════
#  Aspect-preserving size calculation
# ═══════════════════════════════════════════════════════════════════════


def _calc_size(
    cam_w: int, cam_h: int, term_w: int, term_h: int, max_cols: int
) -> tuple[int, int]:
    """Return (cols, halfblock_rows) preserving camera aspect ratio."""
    aspect = cam_w / cam_h
    w = min(term_w, max_cols)
    h = int(w / (aspect * 2))
    if h > term_h:
        h = term_h
        w = int(h * 2 * aspect)
    return max(4, w), max(2, h)


# ═══════════════════════════════════════════════════════════════════════
#  CLI + main loop
# ═══════════════════════════════════════════════════════════════════════


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TUI Webcam — Rich-based demo")
    p.add_argument(
        "--device", type=int, default=0, help="Camera device index (default 0)"
    )
    p.add_argument(
        "--mode", choices=MODES, default="16", help="Initial render mode (default 16)"
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=120,
        help="Maximum render width in columns (default 120)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Cannot open camera (index {args.device})")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    grabber = CameraGrabber(cap)
    while grabber.get() is None:  # wait for first frame
        time.sleep(0.01)

    console = Console()
    camera = CameraFrame()

    mode_idx = MODES.index(args.mode)
    mode = MODES[mode_idx]
    max_w = args.max_width
    fps = 0.0

    # Raw terminal mode for keyboard input (Rich does not handle keys).
    fd = sys.stdin.fileno()
    orig = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def _key() -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    try:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                t0 = time.monotonic()

                # ── Input ──────────────────────────────────────
                key = _key()
                if key == "q":
                    break
                elif key == "m":
                    mode_idx = (mode_idx + 1) % len(MODES)
                    mode = MODES[mode_idx]

                # ── Frame ──────────────────────────────────────
                frame = grabber.get()
                if frame is None:
                    break

                # ── Render size (480p, aspect-preserved) ──────
                cam_h, cam_w = frame.shape[:2]
                term_w = console.width
                term_h = console.height
                avail_h = max(2, term_h - 2)  # room for status
                rw, rh = _calc_size(cam_w, cam_h, term_w, avail_h, max_w)

                # ── Render ─────────────────────────────────────
                raw_kb = camera.update(frame, rw, rh, mode) / 1024

                # ── Status bar ─────────────────────────────────
                status = Text.assemble(
                    ("  ", ""),
                    (f" {_LABELS[mode]} ", "bold white on dark_green"),
                    (f"  {_BPP[mode]} B/px ", "white on grey23"),
                    (f"  {rw}\u00d7{rh*2}px ", "white on grey23"),
                    (f"  {raw_kb:.0f} kB/fr ", "white on grey23"),
                    (f"  FPS {fps:4.0f} ", "bold yellow on grey23"),
                    ("  [m] mode  [q] quit ", "dim white on grey23"),
                )

                # ── Compose ────────────────────────────────────
                from rich.console import Group

                live.update(Group(camera, status))
                live.refresh()

                dt = time.monotonic() - t0
                fps = 1.0 / max(dt, 1e-9)

                # Throttle to ~30 fps
                remain = 1.0 / 30 - dt
                if remain > 0:
                    time.sleep(remain)

    finally:
        grabber.stop()
        cap.release()
        termios.tcsetattr(fd, termios.TCSADRAIN, orig)


if __name__ == "__main__":
    main()
