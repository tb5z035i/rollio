"""Terminal frame renderer with multiple colour-depth modes.

Renders BGR numpy frames as half-block ANSI art.
CPU-only, no CuPy dependency.

Available modes (use the ``mode`` parameter of :func:`render_frame`):

    ===== =========== ============
    Mode  Label       Bytes / px
    ===== =========== ============
    256   256-colour  23
    16    16-colour   13
    gray  grayscale   23
    2     2-colour     3
    ===== =========== ============
"""
from __future__ import annotations

import numpy as np

# ── LUTs ───────────────────────────────────────────────────────────────

_D3 = np.empty((256, 3), dtype=np.uint8)
for _i in range(256):
    _D3[_i] = list(f"{_i:03d}".encode())

# 256-colour template: \x1b[38;5;NNN;48;5;NNNm▀
_C256_TMPL = np.frombuffer(
    b"\x1b[38;5;000;48;5;000m\xe2\x96\x80", dtype=np.uint8).copy()
_C256_LEN = 23

# 16-colour template: \x1b[FFF;BBBm▀
_16_TMPL = np.frombuffer(
    b"\x1b[000;000m\xe2\x96\x80", dtype=np.uint8).copy()
_16_LEN = 13

_CUBE_V = np.array([0, 95, 135, 175, 215, 255], dtype=np.int32)
_CUBE_T = np.array([0, 48, 115, 155, 195, 235], dtype=np.int32)

_ANSI16_RGB = np.array([
    [  0,   0,   0], [170,   0,   0], [  0, 170,   0], [170, 170,   0],
    [  0,   0, 170], [170,   0, 170], [  0, 170, 170], [170, 170, 170],
    [ 85,  85,  85], [255,  85,  85], [ 85, 255,  85], [255, 255,  85],
    [ 85,  85, 255], [255,  85, 255], [ 85, 255, 255], [255, 255, 255],
], dtype=np.int32)

_FG16 = np.empty((16, 3), dtype=np.uint8)
for _i, _c in enumerate(
        [30, 31, 32, 33, 34, 35, 36, 37, 90, 91, 92, 93, 94, 95, 96, 97]):
    _FG16[_i] = list(f"{_c:03d}".encode())

_BG16 = np.empty((16, 3), dtype=np.uint8)
for _i, _c in enumerate(
        [40, 41, 42, 43, 44, 45, 46, 47,
         100, 101, 102, 103, 104, 105, 106, 107]):
    _BG16[_i] = list(f"{_c:03d}".encode())

_2BIT_CHARS = np.array([
    [0xe2, 0xa0, 0x80],   # ⠀  (both dark)
    [0xe2, 0x96, 0x80],   # ▀  (top bright)
    [0xe2, 0x96, 0x84],   # ▄  (bot bright)
    [0xe2, 0x96, 0x88],   # █  (both bright)
], dtype=np.uint8)

_RST_NL = b"\x1b[0m\n"
_RST    = b"\x1b[0m"

# ── Public constants ──────────────────────────────────────────────────

RENDER_MODES: tuple[str, ...] = ("256", "16", "gray", "2")
"""Available render mode identifiers, ordered by decreasing quality."""

MODE_LABELS: dict[str, str] = {
    "256": "256-clr", "16": "16-clr", "gray": "gray", "2": "2-clr",
}
MODE_BPP: dict[str, int] = {"256": 23, "16": 13, "gray": 23, "2": 3}


# ── Colour quantisation helpers ───────────────────────────────────────

def _rgb_to_256(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., c].astype(np.int32) for c in range(3))
    ri = np.clip(np.searchsorted(_CUBE_T, r, side="right") - 1, 0, 5)
    gi = np.clip(np.searchsorted(_CUBE_T, g, side="right") - 1, 0, 5)
    bi = np.clip(np.searchsorted(_CUBE_T, b, side="right") - 1, 0, 5)
    ci = 16 + 36 * ri + 6 * gi + bi
    cd = (r - _CUBE_V[ri])**2 + (g - _CUBE_V[gi])**2 + (b - _CUBE_V[bi])**2
    avg = (r + g + b) // 3
    gri = np.clip((avg - 8 + 5) // 10, 0, 23)
    gv = 8 + gri * 10
    gd = (r - gv)**2 + (g - gv)**2 + (b - gv)**2
    return np.where(gd < cd, 232 + gri, ci).astype(np.uint8)


def _nearest_16(rgb: np.ndarray) -> np.ndarray:
    """Map RGB pixels to the nearest ANSI 16-colour index."""
    flat = rgb.reshape(-1, 3).astype(np.float64)
    ref = _ANSI16_RGB.astype(np.float64)
    ff = (flat ** 2).sum(axis=1)
    rr = (ref ** 2).sum(axis=1)
    fr = flat @ ref.T
    dist = ff[:, None] + rr[None, :] - 2.0 * fr
    return dist.argmin(axis=1).astype(np.uint8).reshape(rgb.shape[:2])


# ── Per-mode pixel-buffer builders ────────────────────────────────────

def _build_256(rgb: np.ndarray, w: int, h: int) -> bytes:
    idx = _rgb_to_256(rgb)
    top = idx[0::2]; bot = idx[1::2]
    n = w * h
    buf = np.tile(_C256_TMPL, n).reshape(n, _C256_LEN)
    buf[:,  7:10] = _D3[top.ravel()]
    buf[:, 16:19] = _D3[bot.ravel()]
    rows = buf.reshape(h, w * _C256_LEN)
    return _RST_NL.join(rows[y].tobytes() for y in range(h)) + _RST


def _build_16(rgb: np.ndarray, w: int, h: int) -> bytes:
    idx = _nearest_16(rgb)
    top = idx[0::2]; bot = idx[1::2]
    n = w * h
    buf = np.tile(_16_TMPL, n).reshape(n, _16_LEN)
    buf[:, 2:5] = _FG16[top.ravel()]
    buf[:, 6:9] = _BG16[bot.ravel()]
    rows = buf.reshape(h, w * _16_LEN)
    return _RST_NL.join(rows[y].tobytes() for y in range(h)) + _RST


def _build_gray(rgb: np.ndarray, w: int, h: int) -> bytes:
    lum = rgb.mean(axis=2).astype(np.int32)
    idx = (np.clip((lum - 8 + 5) // 10, 0, 23) + 232).astype(np.uint8)
    top = idx[0::2]; bot = idx[1::2]
    n = w * h
    buf = np.tile(_C256_TMPL, n).reshape(n, _C256_LEN)
    buf[:,  7:10] = _D3[top.ravel()]
    buf[:, 16:19] = _D3[bot.ravel()]
    rows = buf.reshape(h, w * _C256_LEN)
    return _RST_NL.join(rows[y].tobytes() for y in range(h)) + _RST


def _build_2(rgb: np.ndarray, w: int, h: int) -> bytes:
    lum = rgb.mean(axis=2)
    bright = (lum > lum.mean()).astype(np.uint8)
    top = bright[0::2]; bot = bright[1::2]
    idx = top * 2 + bot
    flat = _2BIT_CHARS[idx.ravel()]
    rows = flat.reshape(h, w * 3)
    prefix = b"\x1b[97;40m"
    return _RST_NL.join(
        prefix + rows[y].tobytes() for y in range(h)) + _RST


_BUILDERS: dict[str, object] = {
    "256":  _build_256,
    "16":   _build_16,
    "gray": _build_gray,
    "2":    _build_2,
}


# ── Public API ────────────────────────────────────────────────────────

def render_frame(bgr: np.ndarray, w: int, h: int,
                 mode: str = "256") -> bytes:
    """Render a BGR frame to half-block ANSI bytes.

    The frame is resized to *w* columns x *h* x 2 pixel rows (half-block
    encoding packs two pixel rows per character row).

    Parameters
    ----------
    bgr : np.ndarray
        Input image in BGR colour order (OpenCV default).
    w, h : int
        Output width in columns and height in half-block rows.
    mode : str
        Colour depth — one of :data:`RENDER_MODES`.

    Returns raw bytes ready for ``sys.stdout.buffer.write()``.
    """
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h * 2), interpolation=cv2.INTER_AREA)
    builder = _BUILDERS.get(mode)
    if builder is None:
        raise ValueError(
            f"Unknown render mode {mode!r}; choose from {RENDER_MODES}")
    return builder(rgb, w, h)


def calc_render_size(cam_w: int, cam_h: int,
                     avail_w: int, avail_h: int,
                     max_cols: int = 120) -> tuple[int, int]:
    """Calculate render size preserving the camera's aspect ratio.

    Terminal character cells are roughly twice as tall as wide.  The
    half-block technique maps two pixel rows per character row, making
    effective pixels approximately square.

    Returns ``(width_cols, height_halfblock_rows)``.
    """
    aspect = cam_w / cam_h
    w = min(avail_w, max_cols)
    h = int(w / (aspect * 2))
    if h > avail_h:
        h = avail_h
        w = int(h * 2 * aspect)
    return max(4, w), max(2, h)


def blit_frame(rendered: bytes, start_row: int, start_col: int) -> bytes:
    """Position rendered frame at specific screen coordinates.

    Takes the output of :func:`render_frame` and wraps each row with
    cursor-positioning escapes so the image appears at
    (*start_row*, *start_col*) in the terminal.
    """
    buf = bytearray()
    for i, line in enumerate(rendered.split(_RST_NL)):
        if not line or line == _RST:
            continue
        buf.extend(f"\x1b[{start_row + i};{start_col}H".encode())
        buf.extend(line)
        buf.extend(_RST)
    return bytes(buf)
