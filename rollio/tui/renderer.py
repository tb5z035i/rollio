"""Minimal terminal frame renderer (extracted from tui_webcam.py).

Renders BGR numpy frames as 256-colour half-block ANSI art.
CPU-only, no CuPy dependency.
"""
from __future__ import annotations

import numpy as np

# ── LUTs ───────────────────────────────────────────────────────────────

_D3 = np.empty((256, 3), dtype=np.uint8)
for _i in range(256):
    _D3[_i] = list(f"{_i:03d}".encode())

_C256_TMPL = np.frombuffer(
    b"\x1b[38;5;000;48;5;000m\xe2\x96\x80", dtype=np.uint8).copy()
_C256_LEN = 23

_CUBE_V = np.array([0, 95, 135, 175, 215, 255], dtype=np.int32)
_CUBE_T = np.array([0, 48, 115, 155, 195, 235], dtype=np.int32)

_RST_NL = b"\x1b[0m\n"
_RST    = b"\x1b[0m"


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


def render_frame(bgr: np.ndarray, w: int, h: int) -> bytes:
    """Render a BGR frame to 256-colour half-block ANSI bytes.

    The frame is resized to *w* columns × *h*×2 pixel rows (half-block
    encoding packs 2 rows per character row).

    Returns raw bytes ready for ``sys.stdout.buffer.write()``.
    """
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h * 2), interpolation=cv2.INTER_AREA)

    idx = _rgb_to_256(rgb)
    top = idx[0::2]
    bot = idx[1::2]
    n = w * h

    buf = np.tile(_C256_TMPL, n).reshape(n, _C256_LEN)
    buf[:,  7:10] = _D3[top.ravel()]
    buf[:, 16:19] = _D3[bot.ravel()]

    rows = buf.reshape(h, w * _C256_LEN)
    return _RST_NL.join(rows[y].tobytes() for y in range(h)) + _RST
