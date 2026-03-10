"""Terminal frame renderer with multiple colour-depth modes.

Renders BGR numpy frames as half-block ANSI art.
CPU-only, no CuPy dependency.

Available modes (use the ``mode`` parameter of :func:`render_frame`):

    ===== =========== ============
    Mode  Label       Bytes / px
    ===== =========== ============
    true  24-bit      39
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

# 24-bit truecolour template: \x1b[38;2;RRR;GGG;BBB;48;2;RRR;GGG;BBBm▀
_TC_TMPL = np.frombuffer(
    b"\x1b[38;2;000;000;000;48;2;000;000;000m\xe2\x96\x80", dtype=np.uint8).copy()
_TC_LEN = 39

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

# ── Turbo colormap for depth visualization ────────────────────────────
# Standard turbo: blue (low) → red (high)
_TURBO_BLUE_RED = np.array([
    [48,18,59],[50,21,67],[51,24,74],[52,27,81],[53,30,88],[54,33,95],
    [55,36,102],[56,39,109],[57,42,115],[58,45,121],[59,47,128],[60,50,134],
    [61,53,139],[62,56,145],[63,59,151],[63,62,156],[64,64,162],[65,67,167],
    [65,70,172],[66,73,177],[66,75,181],[67,78,186],[68,81,191],[68,84,195],
    [68,86,199],[69,89,203],[69,92,207],[69,94,211],[70,97,214],[70,100,218],
    [70,102,221],[70,105,224],[70,107,227],[71,110,230],[71,113,233],[71,115,235],
    [71,118,238],[71,120,240],[71,123,242],[70,125,244],[70,128,246],[70,130,248],
    [70,133,250],[70,135,251],[69,138,252],[69,140,253],[68,143,254],[67,145,254],
    [66,148,255],[65,150,255],[64,153,255],[62,155,254],[61,158,254],[59,160,253],
    [58,163,252],[56,165,251],[55,168,250],[53,171,248],[51,173,247],[49,175,245],
    [47,178,244],[46,180,242],[44,183,240],[42,185,238],[40,188,235],[39,190,233],
    [37,192,231],[35,195,228],[34,197,226],[32,199,223],[31,201,221],[30,203,218],
    [28,205,216],[27,208,213],[26,210,210],[26,212,208],[25,213,205],[24,215,202],
    [24,217,200],[24,219,197],[24,221,194],[24,222,192],[24,224,189],[25,226,187],
    [25,227,185],[26,228,182],[28,230,180],[29,231,178],[31,233,175],[32,234,172],
    [34,235,170],[37,236,167],[39,238,164],[42,239,161],[44,240,158],[47,241,155],
    [50,242,152],[53,243,148],[56,244,145],[60,245,142],[63,246,138],[67,247,135],
    [70,248,132],[74,248,128],[78,249,125],[82,250,122],[85,250,118],[89,251,115],
    [93,252,111],[97,252,108],[101,253,105],[105,253,102],[109,254,98],[113,254,95],
    [117,254,92],[121,254,89],[125,255,86],[128,255,83],[132,255,81],[136,255,78],
    [139,255,75],[143,255,73],[146,255,71],[150,254,68],[153,254,66],[156,254,64],
    [159,253,63],[161,253,61],[164,252,60],[167,252,58],[169,251,57],[172,251,56],
    [175,250,55],[177,249,54],[180,248,54],[183,247,53],[185,246,53],[188,245,52],
    [190,244,52],[193,243,52],[195,241,52],[198,240,52],[200,239,52],[203,237,52],
    [205,236,52],[208,234,52],[210,233,53],[212,231,53],[215,229,53],[217,228,54],
    [219,226,54],[221,224,55],[223,223,55],[225,221,55],[227,219,56],[229,217,56],
    [231,215,57],[233,213,57],[235,211,57],[236,209,58],[238,207,58],[239,205,58],
    [241,203,58],[242,201,58],[244,199,58],[245,197,58],[246,195,58],[247,193,58],
    [248,190,57],[249,188,57],[250,186,57],[250,184,56],[251,182,55],[252,179,54],
    [252,177,54],[253,174,53],[253,172,52],[254,169,51],[254,167,50],[254,164,49],
    [254,161,48],[254,158,47],[254,155,45],[254,153,44],[254,150,43],[254,147,42],
    [254,144,41],[253,141,39],[253,138,38],[252,135,37],[252,132,35],[251,129,34],
    [251,126,33],[250,123,31],[249,120,30],[249,117,29],[248,114,28],[247,111,26],
    [246,108,25],[245,105,24],[244,102,23],[243,99,21],[242,96,20],[241,93,19],
    [240,91,18],[239,88,17],[237,85,16],[236,83,15],[235,80,14],[234,78,13],
    [232,75,12],[231,73,12],[229,71,11],[228,69,10],[226,67,10],[225,65,9],
    [223,63,8],[221,61,8],[220,59,7],[218,57,7],[216,55,6],[214,53,6],
    [212,51,5],[210,49,5],[208,47,5],[206,45,4],[204,43,4],[202,42,4],
    [200,40,3],[197,38,3],[195,37,3],[193,35,2],[190,33,2],[188,32,2],
    [185,30,2],[183,29,2],[180,27,1],[178,26,1],[175,24,1],[172,23,1],
    [169,22,1],[167,20,1],[164,19,1],[161,18,1],[158,16,1],[155,15,1],
    [152,14,1],[149,13,1],[146,11,1],[142,10,1],[139,9,2],[136,8,2],
], dtype=np.uint8)

# Reversed turbo for depth: red (close/low) → blue (far/high)
# This is more intuitive: red = close/hot, blue = far/cold
_TURBO_RED_BLUE = _TURBO_BLUE_RED[::-1].copy()

# ── Public constants ──────────────────────────────────────────────────

RENDER_MODES: tuple[str, ...] = ("true", "256", "16", "gray", "2")
"""Available render mode identifiers for RGB frames, ordered by decreasing quality."""

DEPTH_MODES: tuple[str, ...] = ("turbo", "gray")
"""Available render mode identifiers for depth/grayscale frames."""

MODE_LABELS: dict[str, str] = {
    "true": "24-bit", "256": "256-clr", "16": "16-clr", "gray": "gray", "2": "2-clr",
    "turbo": "turbo-cmap",
}
MODE_BPP: dict[str, int] = {"true": 39, "256": 23, "16": 13, "gray": 23, "2": 3, "turbo": 39}


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

def _build_true(rgb: np.ndarray, w: int, h: int) -> bytes:
    """24-bit truecolour: highest quality, highest bandwidth."""
    top = rgb[0::2]
    bot = rgb[1::2]
    n = w * h
    buf = np.tile(_TC_TMPL, n).reshape(n, _TC_LEN)
    tf = top.reshape(n, 3)
    bf = bot.reshape(n, 3)
    # Foreground RGB (top row): positions 7,11,15
    buf[:,  7:10] = _D3[tf[:, 0]]
    buf[:, 11:14] = _D3[tf[:, 1]]
    buf[:, 15:18] = _D3[tf[:, 2]]
    # Background RGB (bottom row): positions 24,28,32
    buf[:, 24:27] = _D3[bf[:, 0]]
    buf[:, 28:31] = _D3[bf[:, 1]]
    buf[:, 32:35] = _D3[bf[:, 2]]
    rows = buf.reshape(h, w * _TC_LEN)
    return _RST_NL.join(rows[y].tobytes() for y in range(h)) + _RST


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
    "true": _build_true,
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
    # Resize before colour conversion so the hot preview loop does not
    # allocate a full-resolution RGB buffer every frame.
    bgr = cv2.resize(bgr, (w, h * 2), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    builder = _BUILDERS.get(mode)
    if builder is None:
        raise ValueError(
            f"Unknown render mode {mode!r}; choose from {RENDER_MODES}")
    return builder(rgb, w, h)


def render_depth(frame: np.ndarray, w: int, h: int,
                 mode: str = "turbo",
                 min_val: float = 0, max_val: float = 0) -> bytes:
    """Render a depth or grayscale frame to half-block ANSI bytes.

    Supports 8-bit and 16-bit single-channel images.

    Parameters
    ----------
    frame : np.ndarray
        Single-channel depth (16-bit) or grayscale (8-bit) image.
    w, h : int
        Output width in columns and height in half-block rows.
    mode : str
        "turbo" for colormap visualization, "gray" for grayscale.
    min_val, max_val : float
        Value range for normalization. If both are 0, auto-range is used.

    Returns raw bytes ready for ``sys.stdout.buffer.write()``.
    """
    import cv2

    # Handle different input formats
    if frame.ndim == 3:
        # Already has channels - convert to grayscale
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = frame[:, :, 0]

    # Resize first
    frame = cv2.resize(frame, (w, h * 2), interpolation=cv2.INTER_AREA)

    # Normalize to 0-255
    if frame.dtype == np.uint16:
        # 16-bit depth - typically in mm, range varies by sensor
        if min_val == 0 and max_val == 0:
            # Auto-range: use 5th-95th percentile for better contrast
            valid = frame[frame > 0]
            if len(valid) > 0:
                min_val = np.percentile(valid, 5)
                max_val = np.percentile(valid, 95)
            else:
                min_val, max_val = 0, 65535
        normalized = np.clip((frame.astype(np.float32) - min_val) /
                             max(1, max_val - min_val) * 255, 0, 255).astype(np.uint8)
    elif frame.dtype == np.uint8:
        # 8-bit grayscale (IR camera)
        normalized = frame
    else:
        # Other types - normalize to 0-255
        if min_val == 0 and max_val == 0:
            min_val, max_val = frame.min(), frame.max()
        normalized = np.clip((frame.astype(np.float32) - min_val) /
                             max(1, max_val - min_val) * 255, 0, 255).astype(np.uint8)

    # Apply colormap or grayscale
    if mode == "turbo":
        # Use red→blue colormap: close (low depth) = red, far (high depth) = blue
        # Scale 0-255 to colormap size
        cmap_idx = (normalized.astype(np.float32) * (len(_TURBO_RED_BLUE) - 1) / 255).astype(np.uint8)
        rgb = _TURBO_RED_BLUE[cmap_idx]
        return _build_true(rgb, w, h)
    else:
        # Grayscale mode
        rgb = np.stack([normalized, normalized, normalized], axis=-1)
        return _build_gray(rgb, w, h)


def normalize_depth_for_display(depth: np.ndarray,
                                 min_mm: float = 200,
                                 max_mm: float = 5000) -> np.ndarray:
    """Normalize 16-bit depth frame to 8-bit for display.

    Parameters
    ----------
    depth : np.ndarray
        16-bit depth image (values in mm).
    min_mm, max_mm : float
        Depth range in mm to map to 0-255.

    Returns 8-bit normalized depth image.
    """
    return np.clip((depth.astype(np.float32) - min_mm) /
                   max(1, max_mm - min_mm) * 255, 0, 255).astype(np.uint8)


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
