#!/usr/bin/env python3
"""
TUI Webcam — GPU-accelerated, delta-rendered terminal webcam viewer.

Seven render modes with different bandwidth / quality trade-offs:
  24-bit  (39 B/px)  →  256-clr (23 B/px)  →  16-clr (13 B/px)
  8-clr   (11 B/px)  →  2-clr   ( 3 B/px)  →  gray   (23 B/px)
  kitty   (PNG ~0.1-0.5 B/px — best quality + lowest bandwidth)

Dependencies:
    pip install opencv-python-headless numpy
    pip install cupy-cuda13x            # optional, for GPU acceleration

Usage:
    python tui_webcam.py                # auto-detect GPU
    python tui_webcam.py --cpu          # force CPU-only mode
    python tui_webcam.py --mode 16      # start in 16-colour mode
    python tui_webcam.py --device 1     # use camera index 1

Controls:
    q           quit
    m           cycle mode
    + / -       brightness up / down
    . / ,       contrast up / down
    ] / [       saturation up / down
    d           toggle dithering (8/16/2-colour modes)
    t           cycle noise threshold (0 → 2 → 5 → 10 → 0…)
"""

import argparse
import base64
import os
import select
import signal
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np

# ── Early --cpu check (must happen before CuPy import) ────────────────
_FORCE_CPU = "--cpu" in sys.argv

# ── optional GPU back-end ──────────────────────────────────────────────
_HAS_CUPY = False
if not _FORCE_CPU:
    try:
        import cupy as cp
        cp.empty(1)
        _HAS_CUPY = True
    except Exception:
        pass

xp = cp if _HAS_CUPY else np

# ── Kitty graphics protocol detection ─────────────────────────────────
_HAS_KITTY = os.environ.get("TERM_PROGRAM") in ("kitty", "WezTerm")
if not _HAS_KITTY and "kitty" in os.environ.get("TERM", ""):
    _HAS_KITTY = True

# ── Synchronized output (DEC 2026) ────────────────────────────────────
_SYNC_START = b"\x1b[?2026h"
_SYNC_END   = b"\x1b[?2026l"

_KITTY_DELETE_ALL = b"\x1b_Ga=d,d=A,q=2\x1b\\"


# ═══════════════════════════════════════════════════════════════════════════
#  Look-up tables
# ═══════════════════════════════════════════════════════════════════════════

_D3_np = np.empty((256, 3), dtype=np.uint8)
for _i in range(256):
    _D3_np[_i] = list(f"{_i:03d}".encode())
_D3 = xp.asarray(_D3_np)

_TC_TMPL = xp.asarray(np.frombuffer(
    b"\x1b[38;2;000;000;000;48;2;000;000;000m\xe2\x96\x80", dtype=np.uint8))
_TC_LEN = 39

_C256_TMPL = xp.asarray(np.frombuffer(
    b"\x1b[38;5;000;48;5;000m\xe2\x96\x80", dtype=np.uint8))
_C256_LEN = 23

_16_TMPL = xp.asarray(np.frombuffer(
    b"\x1b[000;000m\xe2\x96\x80", dtype=np.uint8))
_16_LEN = 13

_8_TMPL = xp.asarray(np.frombuffer(
    b"\x1b[30;40m\xe2\x96\x80", dtype=np.uint8))
_8_LEN = 11

_2_LEN = 3
_2_PREFIX = b"\x1b[97;40m"
_2BIT_CHARS = xp.array([
    [0xe2, 0xa0, 0x80],
    [0xe2, 0x96, 0x80],
    [0xe2, 0x96, 0x84],
    [0xe2, 0x96, 0x88],
], dtype=xp.uint8)

_CUBE_V = xp.array([0, 95, 135, 175, 215, 255], dtype=xp.int32)
_CUBE_T_np = np.array([0, 48, 115, 155, 195, 235], dtype=np.int32)

_ANSI16_RGB = xp.array([
    [  0,   0,   0], [170,   0,   0], [  0, 170,   0], [170, 170,   0],
    [  0,   0, 170], [170,   0, 170], [  0, 170, 170], [170, 170, 170],
    [ 85,  85,  85], [255,  85,  85], [ 85, 255,  85], [255, 255,  85],
    [ 85,  85, 255], [255,  85, 255], [ 85, 255, 255], [255, 255, 255],
], dtype=xp.int32)
_ANSI8_RGB = _ANSI16_RGB[:8]

_FG16_np = np.empty((16, 3), dtype=np.uint8)
for _i, _c in enumerate([30,31,32,33,34,35,36,37,90,91,92,93,94,95,96,97]):
    _FG16_np[_i] = list(f"{_c:03d}".encode())
_FG16 = xp.asarray(_FG16_np)

_BG16_np = np.empty((16, 3), dtype=np.uint8)
for _i, _c in enumerate([40,41,42,43,44,45,46,47,100,101,102,103,104,105,106,107]):
    _BG16_np[_i] = list(f"{_c:03d}".encode())
_BG16 = xp.asarray(_BG16_np)

_DIGIT8 = xp.array([0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37],
                    dtype=xp.uint8)

_RST_NL = b"\x1b[0m\n"
_RST    = b"\x1b[0m"


# ═══════════════════════════════════════════════════════════════════════════
#  Background camera grabber
# ═══════════════════════════════════════════════════════════════════════════

class CameraGrabber:
    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                break
            with self._lock:
                self._frame = frame

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Image preprocessing  (brightness, contrast, saturation — combined)
# ═══════════════════════════════════════════════════════════════════════════

def _adjust(arr, bri: float, con: float, sat: float, lib=None):
    """Apply brightness / contrast / saturation in one float pass.

    *lib* is the array library (xp or np) — allows reuse for both
    GPU (xp=cupy) and the kitty path (always np).
    """
    if lib is None:
        lib = xp
    if bri == 1.0 and con == 1.0 and sat == 1.0:
        return arr
    f = arr.astype(lib.float32)
    # Contrast: pivot around 128
    if con != 1.0:
        f = 128.0 + (f - 128.0) * con
    # Brightness: simple multiply
    if bri != 1.0:
        f *= bri
    # Saturation: blend toward per-pixel grey
    if sat != 1.0:
        g = f.mean(axis=2, keepdims=True)
        f = g + (f - g) * sat
    return lib.clip(f, 0, 255).astype(lib.uint8)


def _prep(frame: np.ndarray, w: int, h: int,
          bri: float, con: float, sat: float):
    """BGR frame → resized, adjusted RGB on xp (GPU or CPU)."""
    rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_np = cv2.resize(rgb_np, (w, h), interpolation=cv2.INTER_AREA)
    rgb = xp.asarray(rgb_np)
    return _adjust(rgb, bri, con, sat, xp)


# ═══════════════════════════════════════════════════════════════════════════
#  Pixel-buffer builders  (ANSI modes)
# ═══════════════════════════════════════════════════════════════════════════

def _to_cpu(a) -> np.ndarray:
    return cp.asnumpy(a) if _HAS_CUPY else a


def _nearest_colour(rgb, palette):
    flat = rgb.reshape(-1, 3).astype(xp.float64)
    ref  = palette.astype(xp.float64)
    ff   = (flat ** 2).sum(axis=1)
    rr   = (ref  ** 2).sum(axis=1)
    fr   = flat @ ref.T
    dist = ff[:, None] + rr[None, :] - 2.0 * fr
    return dist.argmin(axis=1).astype(xp.uint8).reshape(rgb.shape[:2])


# ── 4×4 Bayer ordered dithering ───────────────────────────────────────
#  Standard 4×4 threshold matrix normalised to [-0.5, 0.4375].
#  Smaller than 8×8 → the pattern is finer and less visible.
_BAYER = xp.asarray(np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
], dtype=np.float32) / 16.0 - 0.5)   # (4, 4) float32

# Dithering enabled state  (toggled with 'd' key)
_dither_on = True

def _dither(rgb, spread: float):
    """Apply 4×4 ordered dithering before colour quantisation.

    *spread* is the peak offset in colour-value units.  Keep it well
    below the palette step to avoid an overpowering checkerboard.
    """
    h, w = rgb.shape[:2]
    tiled = xp.tile(_BAYER, ((h + 3) // 4, (w + 3) // 4))[:h, :w]
    offset = tiled[:, :, None] * xp.float32(spread)
    return xp.clip(rgb.astype(xp.float32) + offset,
                   0, 255).astype(xp.uint8)


def _build_tc(rgb, w, h):
    top = rgb[0::2]; bot = rgb[1::2]
    n = w * h
    buf = xp.tile(_TC_TMPL, n).reshape(n, _TC_LEN)
    tf = top.reshape(n, 3); bf = bot.reshape(n, 3)
    buf[:,  7:10] = _D3[tf[:, 0]]
    buf[:, 11:14] = _D3[tf[:, 1]]
    buf[:, 15:18] = _D3[tf[:, 2]]
    buf[:, 24:27] = _D3[bf[:, 0]]
    buf[:, 28:31] = _D3[bf[:, 1]]
    buf[:, 32:35] = _D3[bf[:, 2]]
    return buf

def _rgb_to_256(rgb):
    r = rgb[..., 0].astype(xp.int32)
    g = rgb[..., 1].astype(xp.int32)
    b = rgb[..., 2].astype(xp.int32)
    r_c, g_c, b_c = _to_cpu(r), _to_cpu(g), _to_cpu(b)
    ri = np.clip(np.searchsorted(_CUBE_T_np, r_c, side="right") - 1, 0, 5)
    gi = np.clip(np.searchsorted(_CUBE_T_np, g_c, side="right") - 1, 0, 5)
    bi = np.clip(np.searchsorted(_CUBE_T_np, b_c, side="right") - 1, 0, 5)
    cv_ = _to_cpu(_CUBE_V)
    ci = 16 + 36 * ri + 6 * gi + bi
    cd = (r_c - cv_[ri])**2 + (g_c - cv_[gi])**2 + (b_c - cv_[bi])**2
    avg = (r_c + g_c + b_c) // 3
    gri = np.clip((avg - 8 + 5) // 10, 0, 23)
    gv  = 8 + gri * 10
    gd  = (r_c - gv)**2 + (g_c - gv)**2 + (b_c - gv)**2
    return xp.asarray(np.where(gd < cd, 232 + gri, ci).astype(np.uint8))

def _build_256(rgb, w, h):
    idx = _rgb_to_256(rgb)
    top = idx[0::2]; bot = idx[1::2]
    n = w * h
    buf = xp.tile(_C256_TMPL, n).reshape(n, _C256_LEN)
    buf[:,  7:10] = _D3[top.ravel()]
    buf[:, 16:19] = _D3[bot.ravel()]
    return buf

def _build_16(rgb, w, h):
    src = _dither(rgb, 15) if _dither_on else rgb     # step≈85, subtle
    idx = _nearest_colour(src, _ANSI16_RGB)
    top = idx[0::2]; bot = idx[1::2]
    n = w * h
    buf = xp.tile(_16_TMPL, n).reshape(n, _16_LEN)
    buf[:, 2:5] = _FG16[top.ravel()]
    buf[:, 6:9] = _BG16[bot.ravel()]
    return buf

def _build_8(rgb, w, h):
    src = _dither(rgb, 25) if _dither_on else rgb     # step≈170, subtle
    idx = _nearest_colour(src, _ANSI8_RGB)
    top = idx[0::2]; bot = idx[1::2]
    n = w * h
    buf = xp.tile(_8_TMPL, n).reshape(n, _8_LEN)
    buf[:, 3] = _DIGIT8[top.ravel()]
    buf[:, 6] = _DIGIT8[bot.ravel()]
    return buf

def _build_2(rgb, w, h):
    lum = rgb.mean(axis=2)
    if _dither_on:
        h2, w2 = lum.shape
        tiled = xp.tile(_BAYER, ((h2 + 3) // 4, (w2 + 3) // 4))[:h2, :w2]
        bright = (lum > lum.mean() + tiled * 25).astype(xp.uint8)
    else:
        bright = (lum > lum.mean()).astype(xp.uint8)
    top = bright[0::2]; bot = bright[1::2]
    idx = top * 2 + bot
    return _2BIT_CHARS[idx.ravel()]

def _build_gray(rgb, w, h):
    lum = rgb.mean(axis=2).astype(xp.int32)
    idx = (xp.clip((lum - 8 + 5) // 10, 0, 23) + 232).astype(xp.uint8)
    top = idx[0::2]; bot = idx[1::2]
    n = w * h
    buf = xp.tile(_C256_TMPL, n).reshape(n, _C256_LEN)
    buf[:,  7:10] = _D3[top.ravel()]
    buf[:, 16:19] = _D3[bot.ravel()]
    return buf


_BUILDERS = {
    "true": (_build_tc,   _TC_LEN,   b""),
    "256":  (_build_256,  _C256_LEN, b""),
    "16":   (_build_16,   _16_LEN,   b""),
    "8":    (_build_8,    _8_LEN,    b""),
    "2":    (_build_2,    _2_LEN,    _2_PREFIX),
    "gray": (_build_gray, _C256_LEN, b""),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Kitty graphics protocol renderer
# ═══════════════════════════════════════════════════════════════════════════

def _kitty_frame(frame_bgr: np.ndarray, w_cols: int, h_rows: int,
                 bri: float, con: float, sat: float) -> tuple[bytes, int]:
    rgb_np = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_np = cv2.resize(rgb_np, (w_cols, h_rows * 2),
                        interpolation=cv2.INTER_AREA)
    rgb_np = _adjust(rgb_np, bri, con, sat, np)

    ok, png_buf = cv2.imencode(".png", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        return b"", 0
    png_bytes = png_buf.tobytes()
    b64 = base64.standard_b64encode(png_bytes)

    header = (f"\x1b_Ga=T,f=100,t=d,"
              f"c={w_cols},r={h_rows},"
              f"z=-1,q=2").encode()

    CHUNK = 4096
    parts = []
    for i in range(0, len(b64), CHUNK):
        chunk = b64[i:i + CHUNK]
        is_last = (i + CHUNK >= len(b64))
        if i == 0 and is_last:
            parts.append(header + b";" + chunk + b"\x1b\\")
        elif i == 0:
            parts.append(header + b",m=1;" + chunk + b"\x1b\\")
        elif is_last:
            parts.append(b"\x1b_Gm=0;" + chunk + b"\x1b\\")
        else:
            parts.append(b"\x1b_Gm=1;" + chunk + b"\x1b\\")

    payload = (_KITTY_DELETE_ALL + b"\x1b[2J\x1b[H"
               + b"".join(parts))
    return payload, len(png_bytes)


# ═══════════════════════════════════════════════════════════════════════════
#  Output generators  (ANSI modes)
# ═══════════════════════════════════════════════════════════════════════════

def _full_output(buf_cpu, w, h, pxlen, prefix=b""):
    rows = buf_cpu.reshape(h, w * pxlen)
    return b"\x1b[H" + _RST_NL.join(
        prefix + rows[y].tobytes() for y in range(h)
    ) + _RST

def _delta_output(buf_cpu, changed_cpu, w, h, pxlen, prefix=b""):
    buf3d = buf_cpu.reshape(h, w, pxlen)
    out = bytearray()
    _pad = np.array([False], dtype=bool)
    for y in range(h):
        mask = changed_cpu[y]
        if not mask.any():
            continue
        padded = np.concatenate((_pad, mask, _pad))
        edges  = np.diff(padded.view(np.int8))
        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0]
        row_y  = y + 1
        for s, e in zip(starts, ends):
            out.extend(f"\x1b[{row_y};{s + 1}H".encode())
            out.extend(prefix)
            out.extend(buf3d[y, s:e].tobytes())
    out.extend(_RST)
    return bytes(out)

def _change_mask(rgb, prev_rgb, h, w, threshold):
    top_c = rgb[0::2];      bot_c = rgb[1::2]
    top_p = prev_rgb[0::2]; bot_p = prev_rgb[1::2]
    if threshold <= 0:
        td = xp.any(top_c != top_p, axis=2)
        bd = xp.any(bot_c != bot_p, axis=2)
    else:
        td = xp.any(xp.abs(top_c.astype(xp.int16) - top_p.astype(xp.int16)) > threshold, axis=2)
        bd = xp.any(xp.abs(bot_c.astype(xp.int16) - bot_p.astype(xp.int16)) > threshold, axis=2)
    return td | bd


# ═══════════════════════════════════════════════════════════════════════════
#  Terminal wrapper
# ═══════════════════════════════════════════════════════════════════════════

class Terminal:
    def __init__(self) -> None:
        self.fd   = sys.stdin.fileno()
        self.orig = None
        self.cols = 80
        self.rows = 24

    def __enter__(self) -> "Terminal":
        self.orig = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        sys.stdout.buffer.write(b"\x1b[?1049h\x1b[?25l")
        sys.stdout.buffer.flush()
        self._update_size()
        signal.signal(signal.SIGWINCH, self._on_resize)
        return self

    def __exit__(self, *_: object) -> None:
        sys.stdout.buffer.write(b"\x1b[?25h\x1b[?1049l")
        sys.stdout.buffer.flush()
        if self.orig is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.orig)

    def _on_resize(self, *_: object) -> None:
        self._update_size()

    def _update_size(self) -> None:
        self.cols, self.rows = os.get_terminal_size()

    def read_key(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  GPU warm-up
# ═══════════════════════════════════════════════════════════════════════════

def _warmup() -> None:
    if not _HAS_CUPY:
        return
    d = xp.zeros((4, 4, 3), dtype=xp.uint8)
    for builder, _, _ in _BUILDERS.values():
        builder(d, 4, 2)
    cp.cuda.Device(0).synchronize()


# ═══════════════════════════════════════════════════════════════════════════
#  CLI + Main loop
# ═══════════════════════════════════════════════════════════════════════════

ALL_MODES = ["kitty", "true", "256", "16", "8", "2", "gray"]
_LABELS   = {"true": "24-bit", "256": "256-clr", "16": "16-clr",
             "8": "8-clr", "2": "2-clr", "gray": "gray", "kitty": "kitty"}
_BPP      = {"true": 39, "256": 23, "16": 13, "8": 11, "2": 3,
             "gray": 23, "kitty": 0}
_THRESH   = [0, 2, 5, 10]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TUI Webcam — terminal webcam viewer with GPU acceleration")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU-only mode (ignore GPU even if available)")
    p.add_argument("--mode", choices=ALL_MODES, default=None,
                   help="Initial render mode (default: kitty if available, "
                        "else true)")
    p.add_argument("--device", type=int, default=0,
                   help="Camera device index (default: 0)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"⚠  Cannot open camera (index {args.device}).")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    backend = "CuPy/GPU" if _HAS_CUPY else "NumPy/CPU"
    gfx     = "kitty-gfx ✓" if _HAS_KITTY else "kitty-gfx ✗"
    print(f"Backend: {backend}  {gfx} — warming up…", end="", flush=True)
    _warmup()
    print(" done.")

    # Build available modes list
    modes = [m for m in ALL_MODES
             if m != "kitty" or _HAS_KITTY]

    # Pick initial mode
    if args.mode:
        if args.mode == "kitty" and not _HAS_KITTY:
            print("⚠  Kitty graphics not available in this terminal, "
                  "falling back to 24-bit.")
            mode = "true"
        else:
            mode = args.mode
    else:
        mode = modes[0]

    grabber = CameraGrabber(cap)
    while grabber.get() is None:
        time.sleep(0.01)

    bri, con, sat = 1.0, 1.0, 1.0
    fps       = 0.0
    thresh_i  = 1
    prev_rgb  = None
    prev_w    = 0
    prev_h    = 0
    prev_mode = mode
    force_full = True

    with Terminal() as term:
        out = sys.stdout.buffer
        try:
            while True:
                t0 = time.monotonic()

                # ── input ────────────────────────────────────────
                key = term.read_key()
                if key == "q":
                    break
                elif key == "m":
                    old_mode = mode
                    mode = modes[(modes.index(mode) + 1) % len(modes)]
                    if old_mode == "kitty":
                        out.write(_KITTY_DELETE_ALL + b"\x1b[2J")
                        out.flush()
                    force_full = True
                elif key in ("+", "="):
                    bri = min(bri + 0.1, 3.0)
                elif key == "-":
                    bri = max(bri - 0.1, 0.1)
                elif key == ".":
                    con = min(con + 0.1, 3.0)
                elif key == ",":
                    con = max(con - 0.1, 0.1)
                elif key == "]":
                    sat = min(sat + 0.1, 3.0)
                elif key == "[":
                    sat = max(sat - 0.1, 0.0)
                elif key == "d":
                    global _dither_on
                    _dither_on = not _dither_on
                    force_full = True
                elif key == "t":
                    thresh_i = (thresh_i + 1) % len(_THRESH)

                # ── grab latest frame ────────────────────────────
                frame = grabber.get()
                if frame is None:
                    break

                w = term.cols
                h = term.rows - 1
                if w < 4 or h < 2:
                    time.sleep(0.05)
                    continue

                threshold = _THRESH[thresh_i]

                # ── Kitty mode ───────────────────────────────────
                if mode == "kitty":
                    payload, png_size = _kitty_frame(
                        frame, w, h, bri, con, sat)
                    t_render = time.monotonic() - t0
                    kB = len(payload) / 1024
                    pct_changed = 100.0
                    bpp_str = f"PNG {png_size/1024:.0f}kB"

                else:
                    # ── ANSI modes ───────────────────────────────
                    rgb = _prep(frame, w, h * 2, bri, con, sat)

                    builder, pxlen, prefix = _BUILDERS[mode]
                    buf = builder(rgb, w, h)
                    buf_cpu = _to_cpu(buf)

                    t_render = time.monotonic() - t0

                    need_full = (force_full
                                 or prev_rgb is None
                                 or prev_w != w or prev_h != h
                                 or prev_mode != mode)

                    if need_full:
                        payload = (b"\x1b[2J"
                                   + _full_output(buf_cpu, w, h, pxlen,
                                                  prefix))
                        force_full = False
                        pct_changed = 100.0
                    else:
                        changed = _change_mask(rgb, prev_rgb, h, w,
                                               threshold)
                        changed_cpu = _to_cpu(changed)
                        n_changed = int(changed_cpu.sum())
                        total = w * h
                        pct_changed = n_changed / total * 100.0
                        if pct_changed > 85.0:
                            payload = _full_output(buf_cpu, w, h, pxlen,
                                                   prefix)
                        else:
                            payload = _delta_output(buf_cpu, changed_cpu,
                                                    w, h, pxlen, prefix)

                    prev_rgb  = rgb.copy()
                    prev_w    = w
                    prev_h    = h
                    prev_mode = mode
                    kB = len(payload) / 1024
                    bpp_str = f"{_BPP[mode]}B/px"

                t_total = time.monotonic() - t0

                # ── status bar ───────────────────────────────────
                bar = (
                    f" FPS {fps:4.0f} │ {_LABELS[mode]} ({bpp_str}) │ "
                    f"bri {bri:.1f} con {con:.1f} sat {sat:.1f} │ "
                    f"{w}×{h*2}px │ {backend} │ "
                    f"render {t_render*1000:.1f}ms │ "
                    f"total {t_total*1000:.1f}ms │ "
                    f"{kB:.0f}kB ({pct_changed:.0f}%) │ "
                    f"thr {threshold} │ "
                    f"dither:{'on' if _dither_on else'off'} │ "
                    f"q m d +/- ./,  [/] t"
                )
                bar = bar[:w].ljust(w)
                bar_bytes = (
                    f"\x1b[{h+1};1H"
                    f"\x1b[48;5;236m\x1b[38;5;250m{bar}\x1b[0m"
                ).encode()

                out.write(_SYNC_START + payload + bar_bytes + _SYNC_END)
                out.flush()

                dt = time.monotonic() - t0
                fps = 1.0 / max(dt, 1e-9)

        finally:
            grabber.stop()
            cap.release()


if __name__ == "__main__":
    main()
